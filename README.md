# tflmfi — Unified Fault-Injection Framework for TFLite-Micro on ESP32.

Framework declarativo para campanhas de injeção de falhas (bit-flip) em modelos TFLite-Micro executados em ESP32-C3. Suporta campanhas de **registradores**, **memória** (pesos do modelo) e **híbridas**, com amostragem estatística estratificada, geração automática de headers C, build/flash via PlatformIO e captura UART.

## Requisitos

- Python >= 3.10
- PlatformIO (para build/flash do firmware)

## Instalação

```bash
pip install -e .
```

## Estrutura do Projeto

```
tflmfi/                  # Pacote Python principal
  cli.py                 # Entry point CLI
  config.py              # Schema YAML e loader de configuração
  population.py          # Introspecção do modelo, geração de população e amostragem
  experiment.py          # Orquestrador: fases opcionais quantize/subset, depois prepare → build → …
  headers.py             # Geração de headers C para o firmware
  builder.py             # Build e flash via PlatformIO
  uart.py                # Captura UART (JSON lines do firmware)
  classify.py            # Classificação dos resultados (Masked, Critical, Crash, etc.)
  analysis.py            # Análise física do modelo (criticidade por tensor, operador, offset)

configs/                 # Exemplos de configuração YAML
  registerfi_example.yaml
  memoryfi_example.yaml
  hybrid_campaign.yaml
  multi_injection_memory_example.yaml
  multi_injection_register_example.yaml

firmware/                # Firmware ESP32 (PlatformIO)
  src/
    main_unified.cpp     # Entrada principal (selecção por macro)
    register_fi.cc/h     # Módulo de injeção em registradores
    memory_fi.cc/h       # Módulo de injeção em memória (pesos do modelo)

tests/                   # Testes (pytest)
models/                  # Modelos .tflite e subsets .npz
runs/                    # Saídas de campanhas (geradas automaticamente)
```

## Tipos de Campanha

### Register (`type: register`)

Injeta bit-flips em registradores RISC-V (x1–x31) durante a execução de operadores do modelo. Usa um timer de hardware para disparar a injeção num instante aleatório dentro da execução do operador.

### Memory (`type: memory`)

Injeta bit-flips nos **pesos constantes do modelo** (tensores serializados no blob `.tflite`). Cada falha altera um bit específico num byte específico de um tensor, identificado por `(op_index, tensor_index, byte_in_tensor, bit_index)`.

### Hybrid (`type: hybrid`)

Executa ambas as campanhas (register + memory) sequencialmente no mesmo target, com artefatos separados por sub-directório.

## Duração da Falha

- **`permanent`** — a falha persiste durante toda a inferência.
- **`transient`** — a falha é aplicada antes do operador alvo e revertida depois (per-operator, via hooks MicroFI).

## Multi-Injection

Suporta injeção de múltiplas falhas simultâneas (simulação MEU — Multi-Event Upset). Grupos aleatórios de `count` falhas são gerados a partir do pool amostrado.

```yaml
multi_injection:
  enabled: true
  count: 2
  num_groups: 500
  seed: 42
```

## Uso

### Pipeline completo

```bash
tflmfi configs/memoryfi_example.yaml
```

### Passos individuais

```bash
tflmfi configs/registerfi_example.yaml --step prepare
tflmfi configs/registerfi_example.yaml --step build
tflmfi configs/registerfi_example.yaml --step flash
tflmfi configs/registerfi_example.yaml --step run
tflmfi configs/registerfi_example.yaml --step analyze
```

### Uso programático

O ciclo principal continua a ser `prepare()` → `build()` → `flash()` → `run()` → `analyze()` (ou `execute()` para tudo de seguida). **Antes** de `prepare()`, o `Experiment` expõe duas fases **opcionais** para gerar os artefactos referenciados no YAML em `model.tflite` e `model.subset_npz`:

| Fase | Método | O que faz |
|------|--------|-----------|
| **0a — Quantização** | `exp.quantize_model(...)` | Converte um modelo Keras (`.h5` / `.keras` / SavedModel) para **TFLite int8** e grava o ficheiro `.tflite`. Por omissão escreve em `cfg.model.tflite` (o mesmo caminho que `prepare()` usa para introspecção e headers). Aceita `dataset_name` (`"mnist"`, `"fashion_mnist"`, `"cifar10"`, `"cifar100"`) para dados de calibração automáticos, ou `representative_data` em NumPy; `preprocess_fn` e `num_calibration_samples` alinham o pré-processamento com o treino. |
| **0b — Subconjunto de teste** | `exp.prepare_test_subset(...)` | Cria um `.npz` aleatório com amostras de teste para o firmware, com as chaves `images`, `labels` e `test_indices` esperadas pelo gerador de headers. Por omissão grava em `cfg.model.subset_npz`. Pode carregar o conjunto de teste via `dataset_name`, `test_data_path` (`.npz` com `(images,labels)`, `(x_test,y_test)` ou `(x,y)`), ou arrays `x`/`y`; `subset_size`, `seed` e `preprocess_fn` controlam o subconjunto e o pré-processamento. |

**Quando usar:** se já tiver um `.tflite` int8 e um `.npz` de teste compatíveis com o YAML, pode ir directamente a `prepare()`. Se parte do pipeline for «do zero» (por exemplo só existe `.h5`), chame primeiro `quantize_model()` e/ou `prepare_test_subset()` para materializar os caminhos configurados; só depois `prepare()` gera população, amostragem e headers C.

```python
from tflmfi import load_config, Experiment

cfg = load_config("configs/memoryfi_example.yaml")
exp = Experiment(cfg)

# Opcional — só se precisar de gerar estes ficheiros a partir do modelo / dados:
# exp.quantize_model("models/meu_modelo.h5", dataset_name="fashion_mnist")
# exp.prepare_test_subset(subset_size=100, dataset_name="fashion_mnist")

exp.prepare()   # introspecção do .tflite + população + amostragem + headers + instalação no firmware
exp.build()     # pio build
exp.flash()     # pio upload
exp.run()       # captura UART
exp.analyze()   # classificação dos resultados
```

## Configuração YAML

Cada experiência é definida por um ficheiro YAML com as seguintes secções:

| Secção | Descrição |
|--------|-----------|
| `experiment` | Nome e descrição |
| `model` | Caminhos para `.tflite` e `.npz` |
| `campaign` | Tipo (`register`/`memory`/`hybrid`), duração, operadores alvo |
| `sampling` | Método estratificado, confiança, margem, seed |
| `firmware` | Directório src, plataforma, board |
| `serial` | Porta, baud rate, timeout, **transporte** (`uart0` ou `usb_serial_jtag`) |
| `output` | Directório de saída, formatos |

Veja os exemplos em `configs/` para referência — em particular `memoryfi_esp32_c6_example.yaml` e `memoryfi_esp32_c3_supermini_example.yaml`.

### Transporte serial (`serial.transport`)

O canal usado pelo firmware para falar com o host depende de **como o conector USB da placa está ligado**:

| Valor | Quando usar | Reset DTR/RTS |
|-------|-------------|---------------|
| `uart0` (default) | Placas com **USB-UART bridge** (CP2102, CH340, FTDI) — ex.: ESP32-C2, DevKits clássicos com chip bridge | Ativado automaticamente |
| `usb_serial_jtag` | Placas em que o USB vai **directo ao controlador USB Serial/JTAG nativo** do chip — ex.: ESP32-C3 Super Mini, C6/H2/P4 DevKitM com USB nativo | Desativado (não funciona sem bridge) |

Quando `transport: usb_serial_jtag`:

- O builder injecta `-DFI_TRANSPORT_USB_SERIAL_JTAG=1` e `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y` no `sdkconfig`, fazendo o firmware usar o driver `usb_serial_jtag` em vez de `UART0`.
- O host salta o toggle DTR/RTS (que não reseta o chip nesse modo) e envia um `{"cmd":"start"}` proativo — o firmware re-emite o evento `boot` a cada ~1 s como rede de segurança.

Campos opcionais relacionados:

```yaml
serial:
  port: COM17
  baud: 460800
  transport: usb_serial_jtag
  run_port: COM17              # porta de captura se for diferente da de flash
  reset_before_capture: null   # null=auto, true=forçar DTR/RTS, false=nunca
  boot_timeout_s: 30
```

> **Como saber qual usar?** Procure o número do chip USB na placa: se encontrar um CP2102, CH340, CH343, FT232 ou similar, é `uart0`; se o conector USB-C/micro-USB vai directo ao SoC (ex.: C3 Super Mini), é `usb_serial_jtag`.

#### Identificar pelo Gerenciador de Dispositivos (Windows)

O nome que o Windows atribui à porta COM é um forte indicador:

| Nome no Gerenciador de Dispositivos | Tipo | `transport` |
|-------------------------------------|------|-------------|
| *"Dispositivo Serial USB"* (driver genérico `usbser.sys`) | USB CDC nativo do SoC | `usb_serial_jtag` |
| *"USB-Enhanced-SERIAL CH343"* / *"USB-SERIAL CH340"* | Bridge WCH CH343/CH340 | `uart0` |
| *"Silicon Labs CP210x USB to UART Bridge"* | Bridge Silicon Labs CP2102/CP2104 | `uart0` |
| *"USB Serial Port"* (driver FTDI) | Bridge FTDI FT232 | `uart0` |

**Exemplos reais testados:**

- `ESP32-C3 Super Mini` → *"Dispositivo Serial USB (COM17)"* → `transport: usb_serial_jtag`
- `ESP32-C6 DevKitC WiFi ZigBee N4` → *"USB-Enhanced-SERIAL CH343 (COM18)"* → `transport: uart0`

## Classificação de Resultados

Cada resultado de inferência com falha é classificado em:

| Classe | Significado |
|--------|-------------|
| **Masked** | Saída idêntica ao golden (sem efeito) |
| **Acceptable** | Mesma classe predita, com variação < 5% na confiança |
| **Good** | Resultado melhor que o golden (corrige uma predição errada) |
| **Observed** | Classe predita diferente do golden, mas correcta e com alta confiança |
| **Warning** | Mesma classe mas com queda significativa de confiança |
| **Critical** | Classe predita errada ou confiança muito baixa |
| **Crash** | Firmware não completou a inferência |

## Análise Física

O módulo `analysis` permite análise espacial da criticidade das falhas:

- Criticidade por tensor e por operador
- Binning por offset físico no blob do modelo
- Heatmap tensor × offset
- Análise de contiguidade entre tensores adjacentes
- Comparação semântica vs. física (Gini / entropia)
- Vulnerabilidade por posição de bit

```python
from tflmfi import generate_physical_report

report = generate_physical_report("runs/my_campaign/")
print(report.summary())
report.save("runs/my_campaign/")
```

## Ambientes PlatformIO

O ficheiro `firmware/platformio.ini` define dois ambientes de build:

- **`registerfi`** — define `FI_CAMPAIGN_REGISTER=1`
- **`memoryfi`** — define `FI_CAMPAIGN_MEMORY=1`

O framework selecciona automaticamente o ambiente correcto com base no `campaign.type`.

## Testes

```bash
pip install -e ".[dev]"
pytest tests/
```
