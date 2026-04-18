#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "driver/uart.h"
#if defined(FI_TRANSPORT_USB_SERIAL_JTAG)
#include "driver/usb_serial_jtag.h"
#endif
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "model_data.h"
#include "mnist_subset.h"

#if defined(FI_CAMPAIGN_REGISTER) && defined(FI_CAMPAIGN_MEMORY)
#error "Define only one campaign: FI_CAMPAIGN_REGISTER or FI_CAMPAIGN_MEMORY"
#endif

#if !defined(FI_CAMPAIGN_REGISTER) && !defined(FI_CAMPAIGN_MEMORY)
#error "Define one campaign: FI_CAMPAIGN_REGISTER or FI_CAMPAIGN_MEMORY"
#endif

#if defined(FI_CAMPAIGN_REGISTER)
#include "register_fault_list.h"
#include "register_fi.h"
#include "register_operator_map.h"
#endif

#if defined(FI_CAMPAIGN_MEMORY)
#include "fault_list.h"
#include "memory_fi.h"
#include "tensor_map.h"
#endif

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "resolver_ops.h"

#if defined(FI_CAMPAIGN_REGISTER)
static RTC_NOINIT_ATTR uint32_t g_rtc_magic;
static RTC_NOINIT_ATTR uint32_t g_rtc_fault_idx;
static RTC_NOINIT_ATTR uint32_t g_rtc_image_id;
static RTC_NOINIT_ATTR uint8_t g_rtc_golden_done;
static RTC_NOINIT_ATTR int32_t g_rtc_golden_pred[128];
static RTC_NOINIT_ATTR float g_rtc_golden_top1[128];
static RTC_NOINIT_ATTR register_fi_op_profile_t g_rtc_op_profiles[32];
static constexpr uint32_t kResumeMagic = 0x52464921u;
#endif

namespace {

#if defined(FI_CAMPAIGN_REGISTER)
constexpr const char* kTag = "registerfi";
#else
constexpr const char* kTag = "memoryfi";
#endif

constexpr size_t kTensorArenaSize = 48 * 1024;
#if defined(FI_CAMPAIGN_REGISTER)
constexpr uint32_t kRegisterTimerMinDelayUs = 25u;
constexpr uint32_t kRegisterTimerGuardUs = 8u;
#endif

alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];
static uint8_t* g_model_ram = nullptr;

#if defined(FI_CAMPAIGN_REGISTER)
static register_fi_context_t g_register_fi_ctx;
#endif

#if defined(FI_CAMPAIGN_MEMORY)
static memory_fi_context_t g_memory_fi_ctx;
#endif

// -------------------------------------------------------------------
// Protocol transport — abstracts UART0 vs USB Serial/JTAG.
// When FI_TRANSPORT_USB_SERIAL_JTAG is defined the firmware talks
// through the chip's built-in USB Serial/JTAG controller (ESP32-C3,
// C5, C6, H2, P4 boards whose USB connector is wired to the native
// USB pins).  Otherwise the classic UART0 path is used.
// -------------------------------------------------------------------
#ifndef FI_PROTOCOL_BAUD
#define FI_PROTOCOL_BAUD 460800
#endif

static char g_proto_buf[2048];

static void protocol_init_uart(void) {
#if defined(FI_TRANSPORT_USB_SERIAL_JTAG)
  usb_serial_jtag_driver_config_t usb_cfg = {
      .tx_buffer_size = 2048,
      .rx_buffer_size = 256,
  };
  ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_cfg));
#else
  const uart_config_t ucfg = {
      .baud_rate  = FI_PROTOCOL_BAUD,
      .data_bits  = UART_DATA_8_BITS,
      .parity     = UART_PARITY_DISABLE,
      .stop_bits  = UART_STOP_BITS_1,
      .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
      .source_clk = UART_SCLK_DEFAULT,
  };
  ESP_ERROR_CHECK(uart_param_config(UART_NUM_0, &ucfg));
  ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 256, 2048, 0, NULL, 0));
#endif
}

static void proto_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(g_proto_buf, sizeof(g_proto_buf), fmt, ap);
  va_end(ap);
  if (n > 0) {
#if defined(FI_TRANSPORT_USB_SERIAL_JTAG)
    usb_serial_jtag_write_bytes(g_proto_buf, n, pdMS_TO_TICKS(1000));
#else
    uart_write_bytes(UART_NUM_0, g_proto_buf, n);
#endif
  }
}

static int proto_read_byte(uint8_t *byte, TickType_t timeout) {
#if defined(FI_TRANSPORT_USB_SERIAL_JTAG)
  return usb_serial_jtag_read_bytes(byte, 1, timeout);
#else
  return uart_read_bytes(UART_NUM_0, byte, 1, timeout);
#endif
}

float dequantize_value(int8_t q, float scale, int32_t zero_point) {
  return (static_cast<int32_t>(q) - zero_point) * scale;
}

struct InferenceSummary {
  int pred_class;
  float top1_prob;
  bool ok;
  const char* injection_status;
#if defined(FI_CAMPAIGN_REGISTER)
  uint32_t delay_us;
  uint8_t register_id;
  const char* register_name;
  uint8_t time_slot_index;
  uint8_t time_slot_count;
#endif
};

InferenceSummary run_inference(tflite::MicroInterpreter* interpreter,
                               const float* image_ptr,
                               uint32_t image_size,
                               int32_t fault_index) {
#if defined(FI_CAMPAIGN_REGISTER)
  REGISTER_FI_SAVE_SP();
#endif

  TfLiteTensor* input = interpreter->input(0);
  if (input == nullptr || input->type != kTfLiteInt8) {
    ESP_LOGE(kTag, "Input tensor is not int8.");
#if defined(FI_CAMPAIGN_REGISTER)
    return {-1, 0.0f, false, "input_not_int8", 0u, 0u, "", 0u, 0u};
#else
    return {-1, 0.0f, false, "input_not_int8"};
#endif
  }

  const float scale = input->params.scale;
  const int32_t zero = input->params.zero_point;
  for (uint32_t i = 0; i < image_size; ++i) {
    int32_t q = static_cast<int32_t>(roundf(image_ptr[i] / scale)) + zero;
    if (q < -128) {
      q = -128;
    }
    if (q > 127) {
      q = 127;
    }
    input->data.int8[i] = static_cast<int8_t>(q);
  }

#if defined(FI_CAMPAIGN_REGISTER)
  uint8_t register_id = 0u;
  const char* register_name = "";
  uint8_t time_slot_index = 0u;
  uint8_t time_slot_count = 0u;

  if (fault_index >= 0 && static_cast<uint32_t>(fault_index) < g_fault_list_count) {
    const register_fault_t* fault = &g_fault_list[fault_index];
    register_id = fault->register_id;
    register_name = register_fi_register_name(fault->register_id);
    time_slot_index = fault->time_slot_index;
    time_slot_count = fault->time_slot_count;
  }

  register_fi_begin_inference(&g_register_fi_ctx, fault_index);
  const TfLiteStatus status = interpreter->Invoke();

  const register_fi_injection_status_t reg_status = register_fi_get_last_status(&g_register_fi_ctx);
  const uint32_t delay_us = register_fi_get_current_delay_us(&g_register_fi_ctx);
  register_fi_end_inference(&g_register_fi_ctx);

  if (status != kTfLiteOk) {
    return {-1, 0.0f, false, register_fi_status_name(reg_status), delay_us, register_id, register_name,
            time_slot_index, time_slot_count};
  }
#endif

#if defined(FI_CAMPAIGN_MEMORY)
  memory_fi_begin_inference(&g_memory_fi_ctx, fault_index);

#if defined(FI_FAULT_TRANSIENT)
  const TfLiteStatus status = interpreter->Invoke();
  const bool was_injected = g_memory_fi_ctx.injected_this_inference;
  memory_fi_end_inference(&g_memory_fi_ctx);

  if (status != kTfLiteOk) {
    return {-1, 0.0f, false, was_injected ? "injected" : "idle"};
  }
#else
  uint8_t* injected_addr = nullptr;
  uint8_t saved_byte = 0u;
  bool injected = false;
  if (fault_index >= 0 && static_cast<uint32_t>(fault_index) < g_memory_fi_ctx.fault_count) {
    const memory_fault_t* fault = &g_memory_fi_ctx.fault_list[fault_index];
    injected_addr = memory_fi_resolve_addr(&g_memory_fi_ctx, fault);
    saved_byte = *injected_addr;
    g_memory_fi_ctx.saved_original_byte = saved_byte;
    *injected_addr ^= static_cast<uint8_t>(1u << fault->bit_index);
    injected = true;
  }

  const TfLiteStatus status = interpreter->Invoke();

  if (injected && injected_addr != nullptr) {
    *injected_addr = saved_byte;
  }

  memory_fi_end_inference(&g_memory_fi_ctx);

  if (status != kTfLiteOk) {
    return {-1, 0.0f, false, injected ? "injected" : "idle"};
  }
#endif
#endif

  TfLiteTensor* output = interpreter->output(0);
  if (output == nullptr || output->type != kTfLiteInt8) {
    ESP_LOGE(kTag, "Output tensor is not int8.");
#if defined(FI_CAMPAIGN_REGISTER)
    return {-1, 0.0f, false, "output_not_int8", 0u, 0u, "", 0u, 0u};
#else
    return {-1, 0.0f, false, "output_not_int8"};
#endif
  }

  const float out_scale = output->params.scale;
  const int32_t out_zero = output->params.zero_point;
  int pred_class = 0;
  float top1 = -1e30f;
  for (int i = 0; i < output->dims->data[output->dims->size - 1]; ++i) {
    const float value = dequantize_value(output->data.int8[i], out_scale, out_zero);
    if (value > top1) {
      top1 = value;
      pred_class = i;
    }
  }

#if defined(FI_CAMPAIGN_REGISTER)
  return {pred_class, top1, true, register_fi_status_name(reg_status), delay_us, register_id, register_name,
          time_slot_index, time_slot_count};
#else
#if defined(FI_FAULT_TRANSIENT)
  return {pred_class, top1, true, was_injected ? "injected" : "idle"};
#else
  const bool fault_selected =
      (fault_index >= 0 && static_cast<uint32_t>(fault_index) < g_memory_fi_ctx.fault_count);
  return {pred_class, top1, true, fault_selected ? "injected" : "idle"};
#endif
#endif
}

}  // namespace

extern "C" {

void protocol_send_boot(void) {
  proto_printf("{\"event\":\"boot\",\"status\":\"ready\"}\n");
}

// Block until the host sends a newline-terminated command over UART.
// Re-emits the boot event every second so the host can recover if the
// original boot was missed (USB re-enumeration, buffer overflow, etc.).
void protocol_wait_for_start(void) {
  char buf[64];
  size_t len = 0;
  while (true) {
    uint8_t byte = 0;
    int n = proto_read_byte(&byte, pdMS_TO_TICKS(1000));
    if (n <= 0) {
      protocol_send_boot();
      continue;
    }
    if (byte == '\n' || byte == '\r') {
      if (len > 0) {
        return;
      }
      continue;
    }
    if (len < sizeof(buf) - 1) {
      buf[len++] = (char)byte;
    }
  }
}

void protocol_send_golden_result(int32_t image_id,
                                 int32_t label,
                                 int32_t pred_class,
                                 float top1_prob,
                                 const char* status) {
  proto_printf(
      "{\"event\":\"result\",\"phase\":\"golden\",\"mode\":\"golden\","
      "\"fault_model\":\"none\",\"fault_id\":-1,\"image_id\":%ld,\"label\":%ld,"
      "\"pred_class\":%ld,\"top1_prob\":%.6f,\"status\":\"%s\",\"injection_status\":\"none\"}\n",
      static_cast<long>(image_id), static_cast<long>(label), static_cast<long>(pred_class), top1_prob, status);
}

#if defined(FI_CAMPAIGN_REGISTER)
void protocol_send_fault_result(int32_t fault_id,
                                int32_t image_id,
                                int32_t label,
                                int32_t pred_class,
                                float top1_prob,
                                int32_t golden_pred_class,
                                float golden_top1_prob,
                                uint32_t operator_index,
                                uint8_t register_id,
                                const char* register_name,
                                uint8_t time_slot_index,
                                uint8_t time_slot_count,
                                uint32_t delay_us,
                                uint8_t bit_index,
                                const char* status,
                                const char* injection_status) {
  proto_printf(
      "{\"event\":\"result\",\"phase\":\"fault\",\"mode\":\"fault\","
      "\"fault_model\":\"register\","
      "\"fault_id\":%ld,\"image_id\":%ld,\"label\":%ld,\"pred_class\":%ld,"
      "\"top1_prob\":%.6f,\"golden_pred_class\":%ld,\"golden_top1_prob\":%.6f,"
      "\"operator_index\":%lu,\"register_id\":%u,\"register_name\":\"%s\","
      "\"time_slot_index\":%u,\"time_slot_count\":%u,\"delay_us\":%lu,\"bit_index\":%u,"
      "\"status\":\"%s\",\"injection_status\":\"%s\"}\n",
      static_cast<long>(fault_id), static_cast<long>(image_id), static_cast<long>(label),
      static_cast<long>(pred_class), top1_prob, static_cast<long>(golden_pred_class),
      golden_top1_prob, static_cast<unsigned long>(operator_index), static_cast<unsigned>(register_id),
      register_name, static_cast<unsigned>(time_slot_index), static_cast<unsigned>(time_slot_count),
      static_cast<unsigned long>(delay_us), static_cast<unsigned>(bit_index), status, injection_status);
}

void protocol_send_operator_profile(uint32_t operator_index,
                                    const char* op_name,
                                    uint32_t mean_duration_us,
                                    uint32_t samples) {
  proto_printf(
      "{\"event\":\"op_profile\",\"fault_model\":\"register\",\"operator_index\":%lu,"
      "\"operator_name\":\"%s\",\"mean_duration_us\":%lu,\"samples\":%lu}\n",
      static_cast<unsigned long>(operator_index), op_name, static_cast<unsigned long>(mean_duration_us),
      static_cast<unsigned long>(samples));
}
#endif

#if defined(FI_CAMPAIGN_MEMORY)
void protocol_send_fault_result(int32_t fault_id,
                                int32_t image_id,
                                int32_t label,
                                int32_t pred_class,
                                float top1_prob,
                                int32_t golden_pred_class,
                                float golden_top1_prob,
                                uint32_t operator_index,
                                uint32_t tensor_index,
                                uint32_t byte_in_tensor,
                                uint8_t bit_index,
                                const char* status,
                                const char* injection_status) {
  proto_printf(
      "{\"event\":\"result\",\"phase\":\"fault\",\"mode\":\"fault\","
      "\"fault_model\":\"memory\","
      "\"fault_id\":%ld,\"image_id\":%ld,\"label\":%ld,\"pred_class\":%ld,"
      "\"top1_prob\":%.6f,\"golden_pred_class\":%ld,\"golden_top1_prob\":%.6f,"
      "\"operator_index\":%lu,\"tensor_index\":%lu,\"byte_in_tensor\":%lu,\"bit_index\":%u,"
      "\"status\":\"%s\",\"injection_status\":\"%s\"}\n",
      static_cast<long>(fault_id), static_cast<long>(image_id), static_cast<long>(label),
      static_cast<long>(pred_class), top1_prob, static_cast<long>(golden_pred_class),
      golden_top1_prob, static_cast<unsigned long>(operator_index), static_cast<unsigned long>(tensor_index),
      static_cast<unsigned long>(byte_in_tensor), static_cast<unsigned>(bit_index), status, injection_status);
}
#endif

#if defined(MULTI_FI_MAX_COUNT)
#if defined(FI_CAMPAIGN_REGISTER)
void protocol_send_multi_fault_result_register(
    int32_t group_id,
    uint8_t fault_count,
    int32_t image_id,
    int32_t label,
    int32_t pred_class,
    float top1_prob,
    int32_t golden_pred_class,
    float golden_top1_prob,
    const char* status,
    const char* injection_status) {
  proto_printf(
      "{\"event\":\"result\",\"phase\":\"fault\",\"mode\":\"multi_fault\","
      "\"fault_model\":\"register\","
      "\"group_id\":%ld,\"fault_count\":%u,\"image_id\":%ld,\"label\":%ld,"
      "\"pred_class\":%ld,\"top1_prob\":%.6f,"
      "\"golden_pred_class\":%ld,\"golden_top1_prob\":%.6f,"
      "\"status\":\"%s\",\"injection_status\":\"%s\"}\n",
      static_cast<long>(group_id), static_cast<unsigned>(fault_count),
      static_cast<long>(image_id), static_cast<long>(label),
      static_cast<long>(pred_class), top1_prob,
      static_cast<long>(golden_pred_class), golden_top1_prob,
      status, injection_status);
}
#endif

#if defined(FI_CAMPAIGN_MEMORY)
void protocol_send_multi_fault_result_memory(
    int32_t group_id,
    uint8_t fault_count,
    int32_t image_id,
    int32_t label,
    int32_t pred_class,
    float top1_prob,
    int32_t golden_pred_class,
    float golden_top1_prob,
    const char* status,
    const char* injection_status) {
  proto_printf(
      "{\"event\":\"result\",\"phase\":\"fault\",\"mode\":\"multi_fault\","
      "\"fault_model\":\"memory\","
      "\"group_id\":%ld,\"fault_count\":%u,\"image_id\":%ld,\"label\":%ld,"
      "\"pred_class\":%ld,\"top1_prob\":%.6f,"
      "\"golden_pred_class\":%ld,\"golden_top1_prob\":%.6f,"
      "\"status\":\"%s\",\"injection_status\":\"%s\"}\n",
      static_cast<long>(group_id), static_cast<unsigned>(fault_count),
      static_cast<long>(image_id), static_cast<long>(label),
      static_cast<long>(pred_class), top1_prob,
      static_cast<long>(golden_pred_class), golden_top1_prob,
      status, injection_status);
}
#endif
#endif  // MULTI_FI_MAX_COUNT

void protocol_send_progress(const char* phase, uint32_t current, uint32_t total) {
  proto_printf("{\"event\":\"progress\",\"phase\":\"%s\",\"current\":%lu,\"total\":%lu}\n",
               phase, static_cast<unsigned long>(current), static_cast<unsigned long>(total));
}

void protocol_send_done(void) {
  proto_printf("{\"event\":\"done\"}\n");
}

}  // extern "C"

extern "C" void app_main(void) {
  protocol_init_uart();
  setvbuf(stdout, NULL, _IOLBF, 256);
  vTaskDelay(pdMS_TO_TICKS(1500));
  protocol_send_boot();
  protocol_wait_for_start();

#if defined(FI_CAMPAIGN_REGISTER)
  const esp_reset_reason_t reset_reason = esp_reset_reason();
  const bool is_crash_reset = (reset_reason == ESP_RST_PANIC || reset_reason == ESP_RST_TASK_WDT ||
                               reset_reason == ESP_RST_INT_WDT || reset_reason == ESP_RST_WDT);
  const bool resuming = (is_crash_reset && g_rtc_magic == kResumeMagic && g_rtc_golden_done == 1u);
  ESP_LOGI(kTag, "Reset reason: %d, resuming: %s", (int)reset_reason, resuming ? "YES" : "NO");
#endif

  g_model_ram = static_cast<uint8_t*>(heap_caps_malloc(g_model_data_len, MALLOC_CAP_8BIT));
  if (g_model_ram == nullptr) {
    ESP_LOGE(kTag, "Failed to allocate model RAM copy.");
    return;
  }
  memcpy(g_model_ram, g_model_data, g_model_data_len);

  const tflite::Model* model = tflite::GetModel(g_model_ram);
  if (model == nullptr) {
    ESP_LOGE(kTag, "Failed to parse TFLite model.");
    return;
  }

  tflite::MicroMutableOpResolver<TFLMFI_RESOLVER_OP_COUNT> resolver;
  tflmfi_populate_resolver(resolver);

  tflite::MicroInterpreter interpreter(model, resolver, g_tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors failed.");
    return;
  }

#if defined(FI_CAMPAIGN_REGISTER)
  register_fi_init(&g_register_fi_ctx, g_fault_list, g_fault_list_count, kRegisterTimerMinDelayUs,
                   kRegisterTimerGuardUs);
#endif

#if defined(FI_CAMPAIGN_MEMORY)
  memory_fi_init(&g_memory_fi_ctx, g_model_ram, g_fault_list, g_fault_list_count);
  ESP_LOGI(kTag, "Tensor map entries: %lu", static_cast<unsigned long>(g_tensor_map_count));
  proto_printf("{\"event\":\"memory_map\","
               "\"model_base\":%lu,\"model_size\":%lu}\n",
               static_cast<unsigned long>(reinterpret_cast<uintptr_t>(g_model_ram)),
               static_cast<unsigned long>(g_model_data_len));
#endif

  const uint32_t image_size = kImageHeight * kImageWidth * kImageChannels;
  static int32_t golden_pred_class[kNumCampaignImages];
  static float golden_top1_prob[kNumCampaignImages];

#if defined(FI_CAMPAIGN_REGISTER)
  uint32_t start_fault_idx = 0u;

  if (resuming) {
    memcpy(golden_pred_class, g_rtc_golden_pred, sizeof(int32_t) * kNumCampaignImages);
    memcpy(golden_top1_prob, g_rtc_golden_top1, sizeof(float) * kNumCampaignImages);
    memcpy(g_register_fi_ctx.op_profiles, g_rtc_op_profiles, sizeof(g_rtc_op_profiles));

    const uint32_t crashed_idx = g_rtc_fault_idx;
    const uint32_t crashed_img = g_rtc_image_id;
#if defined(MULTI_FI_MAX_COUNT)
    if (crashed_idx < g_fault_group_count) {
      const fault_group_t& group = g_fault_groups[crashed_idx];
      for (uint32_t image_id = crashed_img; image_id < kNumCampaignImages; ++image_id) {
        protocol_send_multi_fault_result_register(
            group.group_id, group.fault_count,
            static_cast<int32_t>(image_id), g_mnist_labels[image_id], -1, 0.0f,
            golden_pred_class[image_id], golden_top1_prob[image_id],
            "crash", "crash");
      }
    }
#else
    if (crashed_idx < g_fault_list_count) {
      const register_fault_t& fault = g_fault_list[crashed_idx];
      for (uint32_t image_id = crashed_img; image_id < kNumCampaignImages; ++image_id) {
        protocol_send_fault_result(fault.fault_id, static_cast<int32_t>(image_id), g_mnist_labels[image_id], -1,
                                   0.0f, golden_pred_class[image_id], golden_top1_prob[image_id], fault.op_index,
                                   fault.register_id, register_fi_register_name(fault.register_id),
                                   fault.time_slot_index, fault.time_slot_count, 0u, fault.bit_index, "crash",
                                   "crash");
      }
    }
#endif
    start_fault_idx = crashed_idx + 1u;
  } else {
    g_rtc_magic = kResumeMagic;
    g_rtc_golden_done = 0u;

    register_fi_set_profiling(&g_register_fi_ctx, true);
    protocol_send_progress("golden", 0, kNumCampaignImages);
    for (uint32_t image_id = 0; image_id < kNumCampaignImages; ++image_id) {
      const float* image_ptr = &g_mnist_images[image_id * image_size];
      const InferenceSummary summary = run_inference(&interpreter, image_ptr, image_size, -1);
      golden_pred_class[image_id] = summary.pred_class;
      golden_top1_prob[image_id] = summary.top1_prob;
      protocol_send_golden_result(static_cast<int32_t>(image_id), g_mnist_labels[image_id], summary.pred_class,
                                  summary.top1_prob, summary.ok ? "ok" : "invoke_failed");
      if (image_id % 10 == 0) {
        vTaskDelay(1);
      }
    }
    protocol_send_progress("golden", kNumCampaignImages, kNumCampaignImages);
    register_fi_set_profiling(&g_register_fi_ctx, false);

    memcpy(g_rtc_golden_pred, golden_pred_class, sizeof(int32_t) * kNumCampaignImages);
    memcpy(g_rtc_golden_top1, golden_top1_prob, sizeof(float) * kNumCampaignImages);
    memcpy(g_rtc_op_profiles, g_register_fi_ctx.op_profiles, sizeof(g_rtc_op_profiles));
    g_rtc_golden_done = 1u;

    for (uint32_t i = 0; i < g_register_op_names_count; ++i) {
      const uint32_t op_index = g_register_op_names[i].op_index;
      const uint32_t mean_us = register_fi_get_mean_duration_us(&g_register_fi_ctx, op_index);
      const uint32_t samples = g_register_fi_ctx.op_profiles[op_index].count;
      if (samples == 0u || mean_us == 0u) {
        continue;
      }
      protocol_send_operator_profile(op_index, g_register_op_names[i].op_name, mean_us, samples);
    }
  }

  {
    const esp_task_wdt_config_t twdt_cfg = {
        .timeout_ms = 30000,
        .idle_core_mask = 0,
        .trigger_panic = true,
    };
    esp_task_wdt_reconfigure(&twdt_cfg);
    esp_task_wdt_add(NULL);
  }

#if defined(MULTI_FI_MAX_COUNT)
  for (uint32_t group_idx = start_fault_idx; group_idx < g_fault_group_count; ++group_idx) {
    g_rtc_fault_idx = group_idx;

    if (group_idx % 100 == 0) {
      protocol_send_progress("fault", group_idx, g_fault_group_count);
    }

    const fault_group_t& group = g_fault_groups[group_idx];
    register_fi_set_group(&g_register_fi_ctx, g_fault_pool, group.pool_indices, group.fault_count);

    for (uint32_t image_id = 0; image_id < kNumCampaignImages; ++image_id) {
      g_rtc_image_id = image_id;
      vTaskDelay(1);
      esp_task_wdt_reset();

      memcpy(g_model_ram, g_model_data, g_model_data_len);
      const tflite::Model* fi_model = tflite::GetModel(g_model_ram);
      tflite::MicroInterpreter fi_interp(fi_model, resolver, g_tensor_arena, kTensorArenaSize);
      if (fi_interp.AllocateTensors() != kTfLiteOk) {
        protocol_send_multi_fault_result_register(
            group.group_id, group.fault_count,
            static_cast<int32_t>(image_id), g_mnist_labels[image_id], -1, 0.0f,
            golden_pred_class[image_id], golden_top1_prob[image_id],
            "invoke_failed", "alloc_failed");
        continue;
      }

      const float* image_ptr = &g_mnist_images[image_id * image_size];
      const InferenceSummary summary = run_inference(&fi_interp, image_ptr, image_size, -1);
      protocol_send_multi_fault_result_register(
          group.group_id, group.fault_count,
          static_cast<int32_t>(image_id), g_mnist_labels[image_id],
          summary.pred_class, summary.top1_prob,
          golden_pred_class[image_id], golden_top1_prob[image_id],
          summary.ok ? "ok" : "invoke_failed", summary.injection_status);
    }

    register_fi_clear_group(&g_register_fi_ctx);
  }
#else
  for (uint32_t fault_idx = start_fault_idx; fault_idx < g_fault_list_count; ++fault_idx) {
    g_rtc_fault_idx = fault_idx;

    if (fault_idx % 100 == 0) {
      protocol_send_progress("fault", fault_idx, g_fault_list_count);
    }

    const register_fault_t& fault = g_fault_list[fault_idx];
    for (uint32_t image_id = 0; image_id < kNumCampaignImages; ++image_id) {
      g_rtc_image_id = image_id;
      vTaskDelay(1);
      esp_task_wdt_reset();

      memcpy(g_model_ram, g_model_data, g_model_data_len);
      const tflite::Model* fi_model = tflite::GetModel(g_model_ram);
      tflite::MicroInterpreter fi_interp(fi_model, resolver, g_tensor_arena, kTensorArenaSize);
      if (fi_interp.AllocateTensors() != kTfLiteOk) {
        protocol_send_fault_result(fault.fault_id, static_cast<int32_t>(image_id), g_mnist_labels[image_id], -1,
                                   0.0f, golden_pred_class[image_id], golden_top1_prob[image_id], fault.op_index,
                                   fault.register_id, register_fi_register_name(fault.register_id),
                                   fault.time_slot_index, fault.time_slot_count, 0u, fault.bit_index,
                                   "invoke_failed", "alloc_failed");
        continue;
      }

      const float* image_ptr = &g_mnist_images[image_id * image_size];
      const InferenceSummary summary = run_inference(&fi_interp, image_ptr, image_size, static_cast<int32_t>(fault_idx));
      protocol_send_fault_result(fault.fault_id, static_cast<int32_t>(image_id), g_mnist_labels[image_id],
                                 summary.pred_class, summary.top1_prob, golden_pred_class[image_id],
                                 golden_top1_prob[image_id], fault.op_index, summary.register_id,
                                 summary.register_name, summary.time_slot_index,
                                 summary.time_slot_count, summary.delay_us,
                                 fault.bit_index, summary.ok ? "ok" : "invoke_failed", summary.injection_status);
    }
  }
#endif  // MULTI_FI_MAX_COUNT

  esp_task_wdt_delete(NULL);
  {
    const esp_task_wdt_config_t twdt_cfg = {
        .timeout_ms = 30000,
        .idle_core_mask = (1 << 0),
        .trigger_panic = false,
    };
    esp_task_wdt_reconfigure(&twdt_cfg);
  }

#if defined(MULTI_FI_MAX_COUNT)
  protocol_send_progress("fault", g_fault_group_count, g_fault_group_count);
#else
  protocol_send_progress("fault", g_fault_list_count, g_fault_list_count);
#endif
  protocol_send_done();
  g_rtc_magic = 0u;
  register_fi_deinit(&g_register_fi_ctx);
#endif

#if defined(FI_CAMPAIGN_MEMORY)
  protocol_send_progress("golden", 0, kNumCampaignImages);
  for (uint32_t image_id = 0; image_id < kNumCampaignImages; ++image_id) {
    const float* image_ptr = &g_mnist_images[image_id * image_size];
    const InferenceSummary summary = run_inference(&interpreter, image_ptr, image_size, -1);
    golden_pred_class[image_id] = summary.pred_class;
    golden_top1_prob[image_id] = summary.top1_prob;
    protocol_send_golden_result(static_cast<int32_t>(image_id), g_mnist_labels[image_id], summary.pred_class,
                                summary.top1_prob, summary.ok ? "ok" : "invoke_failed");
    if (image_id % 10 == 0) {
      vTaskDelay(1);
    }
  }
  protocol_send_progress("golden", kNumCampaignImages, kNumCampaignImages);

#if defined(MULTI_FI_MAX_COUNT)
  for (uint32_t group_idx = 0; group_idx < g_fault_group_count; ++group_idx) {
    if (group_idx % 100 == 0) {
      protocol_send_progress("fault", group_idx, g_fault_group_count);
    }

    const fault_group_t& group = g_fault_groups[group_idx];
    memcpy(g_model_ram, g_model_data, g_model_data_len);

#if defined(FI_FAULT_TRANSIENT)
    memory_fi_set_group(&g_memory_fi_ctx, g_fault_pool, group.pool_indices, group.fault_count);

    for (uint32_t image_id = 0; image_id < kNumCampaignImages; ++image_id) {
      const float* image_ptr = &g_mnist_images[image_id * image_size];
      const InferenceSummary summary = run_inference(&interpreter, image_ptr, image_size, -1);

      protocol_send_multi_fault_result_memory(
          group.group_id, group.fault_count,
          static_cast<int32_t>(image_id), g_mnist_labels[image_id],
          summary.pred_class, summary.top1_prob,
          golden_pred_class[image_id], golden_top1_prob[image_id],
          summary.ok ? "ok" : "invoke_failed",
          summary.injection_status);
      if (image_id % 10 == 0) {
        vTaskDelay(1);
      }
    }

    memory_fi_clear_group(&g_memory_fi_ctx);
#else
    memory_fi_set_group(&g_memory_fi_ctx, g_fault_pool, group.pool_indices, group.fault_count);

    for (uint32_t image_id = 0; image_id < kNumCampaignImages; ++image_id) {
      for (uint8_t fi = 0; fi < g_memory_fi_ctx.group_fault_count; ++fi) {
        const memory_fault_t* fault = g_memory_fi_ctx.group_faults[fi];
        uint8_t* addr = memory_fi_resolve_addr(&g_memory_fi_ctx, fault);
        g_memory_fi_ctx.saved_group_bytes[fi] = *addr;
        g_memory_fi_ctx.group_injection_active[fi] = true;
      }
      for (uint8_t fi = 0; fi < g_memory_fi_ctx.group_fault_count; ++fi) {
        const memory_fault_t* fault = g_memory_fi_ctx.group_faults[fi];
        uint8_t* addr = memory_fi_resolve_addr(&g_memory_fi_ctx, fault);
        *addr ^= static_cast<uint8_t>(1u << fault->bit_index);
      }

      const float* image_ptr = &g_mnist_images[image_id * image_size];
      const InferenceSummary summary = run_inference(&interpreter, image_ptr, image_size, -1);

      for (int8_t fi = static_cast<int8_t>(g_memory_fi_ctx.group_fault_count) - 1; fi >= 0; --fi) {
        uint8_t* addr = memory_fi_resolve_addr(&g_memory_fi_ctx, g_memory_fi_ctx.group_faults[fi]);
        *addr = g_memory_fi_ctx.saved_group_bytes[fi];
        g_memory_fi_ctx.group_injection_active[fi] = false;
      }

      protocol_send_multi_fault_result_memory(
          group.group_id, group.fault_count,
          static_cast<int32_t>(image_id), g_mnist_labels[image_id],
          summary.pred_class, summary.top1_prob,
          golden_pred_class[image_id], golden_top1_prob[image_id],
          summary.ok ? "ok" : "invoke_failed",
          summary.ok ? "injected" : "invoke_failed");
      if (image_id % 10 == 0) {
        vTaskDelay(1);
      }
    }

    memory_fi_clear_group(&g_memory_fi_ctx);
#endif
  }

  protocol_send_progress("fault", g_fault_group_count, g_fault_group_count);
#else
  for (uint32_t fault_idx = 0; fault_idx < g_fault_list_count; ++fault_idx) {
    if (fault_idx % 100 == 0) {
      protocol_send_progress("fault", fault_idx, g_fault_list_count);
    }

    const memory_fault_t& fault = g_fault_list[fault_idx];
    memcpy(g_model_ram, g_model_data, g_model_data_len);

    for (uint32_t image_id = 0; image_id < kNumCampaignImages; ++image_id) {
      const float* image_ptr = &g_mnist_images[image_id * image_size];
      const InferenceSummary summary = run_inference(&interpreter, image_ptr, image_size, static_cast<int32_t>(fault_idx));
      protocol_send_fault_result(fault.fault_id, static_cast<int32_t>(image_id), g_mnist_labels[image_id],
                                 summary.pred_class, summary.top1_prob, golden_pred_class[image_id],
                                 golden_top1_prob[image_id], fault.op_index, fault.tensor_index, fault.byte_in_tensor,
                                 fault.bit_index, summary.ok ? "ok" : "invoke_failed", summary.injection_status);
      if (image_id % 10 == 0) {
        vTaskDelay(1);
      }
    }
  }

  protocol_send_progress("fault", g_fault_list_count, g_fault_list_count);
#endif  // MULTI_FI_MAX_COUNT
  protocol_send_done();
#endif
}
