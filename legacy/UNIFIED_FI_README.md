# Unified FI (Register + Memory)

Este rework coloca os dois tipos de campanha no mesmo firmware base, com seleção por macro de build.

## 1) Firmware único

Base usada: `Register_FI/Register_FI`

- Entrada principal unificada: `Register_FI/Register_FI/src/main_unified.cpp`
- Módulos incluídos no build:
  - `register_fi.cc` (campanha de registradores)
  - `memory_fi.cc` (campanha de memória)

### Ambientes PlatformIO

Arquivo: `Register_FI/Register_FI/platformio.ini`

- `registerfi` -> define `FI_CAMPAIGN_REGISTER=1`
- `memoryfi` -> define `FI_CAMPAIGN_MEMORY=1`
- `esp32-c3-devkitm-1` -> compatibilidade (equivale a register)

## 2) Geração unificada de população e headers

Novo script:

- `train_convert_SFI_unified/generate_campaign_assets.py`
- Por padrão, ele também instala automaticamente os headers no firmware em `Register_FI/Register_FI/src/`

### Register

```bash
python train_convert_SFI_unified/generate_campaign_assets.py \
  --campaign register \
  --tflite <caminho_modelo_int8.tflite> \
  --subset <caminho_mnist_subset.npz> \
  --out-dir <saida_register>
```

Gera (entre outros):

- `register_population.csv`
- `selected_register_faults.csv`
- `headers_registerfi/register_fault_list.h`
- `headers_registerfi/register_operator_map.h`
- `headers_registerfi/model_data.h`
- `headers_registerfi/mnist_subset.h`

E instala automaticamente em `Register_FI/Register_FI/src/`:

- `model_data.h`
- `mnist_subset.h`
- `register_fault_list.h`
- `register_operator_map.h`

### Memory

```bash
python train_convert_SFI_unified/generate_campaign_assets.py \
  --campaign memory \
  --tflite <caminho_modelo_int8.tflite> \
  --subset <caminho_mnist_subset.npz> \
  --out-dir <saida_memory>
```

Gera (entre outros):

- `memory_population.csv`
- `selected_faults.csv`
- `headers_memoryfi/fault_list.h`
- `headers_memoryfi/tensor_map.h`
- `headers_memoryfi/model_data.h`
- `headers_memoryfi/mnist_subset.h`

E instala automaticamente em `Register_FI/Register_FI/src/`:

- `model_data.h`
- `mnist_subset.h`
- `fault_list.h`
- `tensor_map.h`

## 3) Opções úteis do gerador

- `--firmware-src <path>`: muda o destino de instalação automática dos headers.
- `--no-install`: gera artefatos sem copiar para `src/`.

Exemplo sem instalação automática:

```bash
python train_convert_SFI_unified/generate_campaign_assets.py \
  --campaign register \
  --tflite <caminho_modelo_int8.tflite> \
  --subset <caminho_mnist_subset.npz> \
  --out-dir <saida_register> \
  --no-install
```

## 4) Host unificado

Novos scripts:

- `host_unified/host/uart_receiver.py`
- `host_unified/host/classify_results.py`

Eles aceitam os dois formatos de campanha e geram CSV com colunas unificadas.
