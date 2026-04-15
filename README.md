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
  experiment.py          # Orquestrador do ciclo prepare → build → flash → run → analyze
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

```python
from tflmfi import load_config, Experiment

cfg = load_config("configs/memoryfi_example.yaml")
exp = Experiment(cfg)

exp.prepare()   # população + amostragem + headers
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
| `serial` | Porta, baud rate, timeout |
| `output` | Directório de saída, formatos |

Veja os exemplos em `configs/` para referência.

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
