# tflmfi — Unified Fault-Injection Framework for TFLite-Micro on ESP32.

Framework declarativo para campanhas de injeção de falhas (bit-flip) em modelos TFLite-Micro executados em ESP32-C3. Suporta campanhas de **registradores**, **memória** (pesos do modelo) e **híbridas**, com amostragem estatística estratificada, geração automática de headers C, build/flash via PlatformIO e captura UART.

## Requisitos

- Python >= 3.10
- Uma placa ESP32 suportada e os respetivos drivers USB/serial a funcionar
- Um ambiente virtual Python recomendado para isolar dependências
- PlatformIO Core para build/flash do firmware

O ponto mais importante para o utilizador final é simples: **depois de instalar os requisitos, o fluxo normal passa a ser editar o YAML e correr `tflmfi ...`**. O utilizador não precisa de escolher manualmente o ambiente `registerfi`/`memoryfi` nem de gerar headers à mão.

## Instalação recomendada

### 1. Criar e ativar um ambiente virtual

**PowerShell (Windows)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**bash (Linux/macOS)**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar o framework

```bash
pip install -e .
```

Isto instala o pacote Python e a CLI `tflmfi`.

### 3. Instalar o perfil certo de dependências

**Opção recomendada para a maioria dos utilizadores**

```bash
pip install -e ".[all]"
```

Este perfil instala:

- dependências Python do framework
- dependências de desenvolvimento
- TensorFlow, para quantização e geração de subsets
- PlatformIO Core, para build e flash do firmware

**Perfis úteis**

- `pip install -e .` — apenas o framework e a CLI
- `pip install -e ".[dev]"` — framework + testes/lint
- `pip install -e ".[all]"` — framework + dev + TensorFlow + PlatformIO

> Nota: drivers USB/serial da placa e componentes de toolchain/SDK descarregados pelo PlatformIO continuam dependentes do sistema operativo.

### 4. Verificação rápida

```bash
tflmfi --help
```

Se o comando acima funcionar, a instalação da CLI está correta.

### 5. Opcional — manter o cache do PlatformIO dentro do projeto

Isto é útil quando quiser evitar dependência do diretório global do utilizador.

**PowerShell**

```powershell
$env:PLATFORMIO_CORE_DIR = "$PWD/firmware/.platformio-core"
```

**bash**

```bash
export PLATFORMIO_CORE_DIR="$PWD/firmware/.platformio-core"
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

## Uso

### O que o utilizador precisa realmente de configurar

Na maioria dos casos, o utilizador só precisa de rever estes campos no YAML:

- `model.tflite` — modelo TFLite a usar
- `model.subset_npz` — subset de teste para o firmware
- `campaign.type` — `register`, `memory` ou `hybrid`
- `serial.port` — porta série da placa
- `serial.transport` — `uart0` ou `usb_serial_jtag`
- `firmware.board` — board PlatformIO, se for diferente do exemplo

O resto do pipeline é tratado pelo framework.

### Fluxo recomendado para o dia a dia

1. Escolher ou copiar um YAML em `configs/`
2. Ajustar `serial.port` e `serial.transport`
3. Executar o pipeline completo

```bash
tflmfi configs/registerfi_example.yaml
```

Esse comando executa automaticamente:

- `prepare()` — gera população, amostragem e headers
- `build()` — compila o firmware
- `flash()` — grava o firmware na placa
- `run()` — captura a UART
- `analyze()` — classifica os resultados

Também escolhe automaticamente o ambiente PlatformIO correto com base em `campaign.type`.

### Passos individuais

Use os passos individuais quando quiser depurar alguma fase específica:

```bash
tflmfi configs/registerfi_example.yaml --step prepare
tflmfi configs/registerfi_example.yaml --step build
tflmfi configs/registerfi_example.yaml --step flash
tflmfi configs/registerfi_example.yaml --step run
tflmfi configs/registerfi_example.yaml --step analyze

# Regerar results.csv a partir de results.jsonl
tflmfi configs/memoryfi_example.yaml --step rebuild-csv --run-dir runs/my_run

# Analisar um run já existente
tflmfi configs/memoryfi_example.yaml --step analyze --run-dir runs/my_run
```

### Quando usar a CLI e quando usar a API Python

- Use a **CLI** para o fluxo normal de campanha
- Use a **API Python** quando quiser integrar o framework em notebooks, scripts próprios ou pipelines maiores

### Uso programático

O ciclo principal é `prepare()` → `build()` → `flash()` → `run()` → `analyze()`.

Se já tiver um `.tflite` int8 e um `.npz` compatíveis com o YAML, pode ir diretamente para `prepare()`. Se ainda só tiver um modelo Keras ou precisar de gerar o subset de teste, use antes as fases opcionais `quantize_model()` e `prepare_test_subset()`.

```python
from tflmfi import load_config, Experiment

cfg = load_config("configs/memoryfi_example.yaml")
exp = Experiment(cfg)

# Opcional — só se precisar de gerar estes ficheiros a partir do modelo / dados:
# exp.quantize_model("models/meu_modelo.h5", dataset_name="fashion_mnist")
# exp.prepare_test_subset(subset_size=100, dataset_name="fashion_mnist")

exp.prepare()
exp.build()
exp.flash()
exp.run()
exp.analyze()
```

Se quiser tudo numa chamada só:

```python
from tflmfi import load_config, Experiment

cfg = load_config("configs/registerfi_example.yaml")
exp = Experiment(cfg)
exp.execute()
```

### Novo comportamento para scripts externos

Se o utilizador ativar o ambiente virtual onde o `tflmfi` foi instalado e executar o script com esse Python, o framework passa a resolver automaticamente:

- o executável do PlatformIO a partir do próprio ambiente ativo
- o `PLATFORMIO_CORE_DIR` local ao projeto de firmware, por omissão em `firmware/.platformio-core`

Isso significa que o utilizador pode criar scripts Python **em qualquer pasta**, desde que:

- o ambiente com `tflmfi` esteja ativo
- o script faça `import tflmfi`
- o YAML usado aponte corretamente para o `firmware/src`

Já não é necessário configurar manualmente `PATH` nem `PLATFORMIO_CORE_DIR` para o fluxo programático normal.

### Exemplo: script fora do repositório

Exemplo de um script guardado noutro diretório, desde que o YAML aponte para os caminhos corretos do projeto:

```python
from pathlib import Path
from tflmfi import load_config, Experiment

config_path = Path(r"c:\Downloads\tinymlfi\framework_jose\configs\registerfi_example.yaml")

cfg = load_config(config_path)
exp = Experiment(cfg)
exp.execute()
```

Exemplo de execução:

```powershell
python C:\meus_scripts\run_registerfi.py
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
