# RIFT-V External Reset Supervisor — Raspberry Pi Pico

## Motivação

Durante uma campanha de injeção de falhas, o RIFT-V insere bit-flips em
registradores e memória do ESP32-C3/C6.  Algumas dessas falhas corrompem
registradores críticos do processador (como `sp` ou `gp`) ou travam o
firmware de forma irrecuperável.

No ESP32-C3/C6, a comunicação serial com o host depende da interface
**USB-Serial-JTAG**, que é implementada em hardware mas exige que o
processador continue executando para atender interrupções USB.
Quando uma falha fatal trava o processador, o USB "morre" — o dispositivo
desaparece da lista de portas COM do Windows e **nenhuma estratégia por
software** (DTR/RTS, BREAK, watchdog) consegue recuperá-lo, pois todas
dependem do próprio canal que acabou de morrer.

Sem intervenção, o operador precisaria **desplug/replugar manualmente** o
cabo USB do ESP32 para cada falha fatal — inviável em campanhas com milhares
de injeções.

## Mecanismo

Um **Raspberry Pi Pico** atua como controlador externo (out-of-band),
conectado ao host PC pela sua própria porta USB (independente do ESP32).
O host envia comandos simples ao Pico, que executa duas ações físicas:

### 1. Power-cycle via relé (principal)

Um módulo relé, controlado pelo **GP3** do Pico, interrompe a linha
**VBUS (5 V)** que alimenta o ESP32.  O relé está no caminho entre um
breakout USB-C e o ESP32:

```
USB-C breakout 5V ──── COM do relé
                       NC do relé ──── 5V/VIN do ESP32
```

Quando o Pico ativa o relé, o 5 V é cortado e o ESP32 desliga
completamente.  Quando desativa, o 5 V retorna e o ESP32 faz boot limpo.
Isso equivale a um **unplug/replug físico** do cabo USB.

### 2. Sinal de disconnect USB via D+ (complementar)

Cortar apenas o 5 V não é suficiente.  As linhas de dados USB (D+ e D-)
permanecem fisicamente conectadas ao PC através do breakout.  Sem energia,
os diodos de proteção ESD internos do ESP32 permitem que uma pequena
corrente flua do PC para o chip pelos pinos D+/D- (**phantom power**).
Isso mantém o USB PHY do ESP32 parcialmente vivo — o PC nunca detecta um
"disconnect" real e, quando o 5 V retorna, não re-enumera o dispositivo.

A solução: o **GP4** do Pico, conectado à linha D+ através de um resistor
de 330 Ω, **puxa D+ para GND** antes de cortar o 5 V.  Isso força o PC
a detectar imediatamente a desconexão USB.  Após restaurar a alimentação,
o Pico libera D+ (Hi-Z) e o USB PHY do ESP32 assume o controle
normalmente, acionando o pull-up interno em D+ e disparando a
re-enumeração.

### 3. Reset via EN/RST (fallback)

O **GP2** do Pico está conectado ao pino **EN/RST** do ESP32 em
configuração open-drain.  Ao puxar EN para GND, o ESP32 reinicia.
Este método é mais rápido que o power-cycle e funciona para travamentos
que não matam o USB.  É usado como fallback quando o power-cycle não é
necessário.

## Por que funciona

O mecanismo funciona porque contorna completamente o canal USB do ESP32:

1. **Independência do canal de controle** — O Pico possui sua própria
   conexão USB com o host.  Mesmo que o ESP32 esteja 100% travado e
   invisível para o SO, o Pico continua respondendo a comandos.

2. **Desconexão USB completa** — A combinação de cortar o 5 V (relé) com
   forçar D+ = 0 V (GP4) simula um unplug físico real.  O PC vê D+ cair,
   registra o disconnect, e remove a porta COM.  Quando a energia retorna
   e D+ é liberado, o ESP32 faz boot limpo, o USB PHY inicializa do zero,
   aciona o pull-up em D+, e o PC re-enumera o dispositivo — a porta COM
   reaparece automaticamente.

3. **Boot limpo garantido** — Durante o power-off, o Pico também segura
   EN/RST em GND.  Isso garante que, ao restaurar o 5 V, o ESP32 não
   inicie enquanto a alimentação ainda está instabilizando.  Só após a
   estabilização o Pico libera EN/RST, permitindo um boot ordenado.

4. **Timing controlado** — A sequência completa é:
   - GP4 puxa D+ LOW (PC vê disconnect)
   - 50 ms de espera
   - Relé corta 5 V
   - Espera configurável (1–10 s, default 3 s)
   - Relé restaura 5 V + 100 ms de estabilização
   - GP4 libera D+ (Hi-Z) → ESP32 USB PHY assume → PC re-enumera
   - EN/RST liberado → ESP32 inicia boot
   - 3 s de espera para boot + enumeração USB completar

## Hardware

### Materiais

| Item                          | Obrigatório |
|-------------------------------|-------------|
| Raspberry Pi Pico (RP2040)    | Sim         |
| Cabo USB para o Pico          | Sim         |
| Módulo relé 1 canal (5 V)     | Sim         |
| Breakout USB-C                | Sim         |
| Resistor 330 Ω (GP4 → D+)    | Sim         |
| Resistor 1 kΩ (GP2 → EN/RST) | Opcional    |
| Jumper wires                  | Sim         |

### Pinagem do Pico

| GPIO  | Pino físico | Função                         |
|-------|-------------|--------------------------------|
| VBUS  | 40          | Alimenta o módulo relé (5 V)   |
| GND   | 38          | Terra comum                    |
| GP2   | 4           | Open-drain EN/RST do ESP32     |
| GP3   | 5           | IN do módulo relé              |
| GP4   | 6           | Força D+ LOW (com 330 Ω)       |

### Diagrama de ligação

```
                         CONTROLE DO RELÉ
                         ────────────────

Raspberry Pi Pico                         Módulo Relé
─────────────────                         ───────────
pino 40 VBUS / 5V  ─────────────────────→ VCC
pino 38 GND        ─────────────────────→ GND
pino 5  GP3        ─────────────────────→ IN


                         USB POWER-CYCLE
                         ───────────────

USB-C Breakout                            Módulo Relé           ESP32-C6
──────────────                            ───────────           ────────
VBUS / 5V      ─────────────────────────→ COM
                                          NC ──────────────────→ 5V / VIN

GND            ────────────────────────────────────────────────→ GND
D+             ──┬─────────────────────────────────────────────→ GPIO13 / USB D+
                 │
                 └── [330Ω] ── Pico GP4 (pino 6)
D-             ────────────────────────────────────────────────→ GPIO12 / USB D-


                         RESET OPCIONAL
                         ──────────────

Raspberry Pi Pico                         ESP32-C6
─────────────────                         ────────
pino 4 GP2        ──[1kΩ opt.]──────────→ EN / RST
GND               ─────────────────────→ GND
```

**Importante:**

- Use **COM** e **NC** do relé.  Não use NO.
- **Não** conecte 3V3 entre as duas placas.
- Mantenha os fios de D+ e D- **curtos e paralelos** (USB é sensível).
- Alguns breakouts marcam `DP`/`DM` em vez de `D+`/`D-`.

## Firmware setup (PlatformIO)

### First upload (Pico em modo BOOTSEL)

```bash
# 1. Segure BOOTSEL no Pico, plugue o USB, solte BOOTSEL.
# 2. A partir da raiz do projeto:
cd supervisor
pio run -t upload
```

### Uploads subsequentes (automático)

```bash
cd supervisor
pio run -t upload
```

PlatformIO reseta o Pico em BOOTSEL automaticamente — sem botão.

### Monitor serial

```bash
pio device monitor
```

## Protocolo

115200 baud, terminado por `\n`:

| Host → Pico          | Pico → Host                | Descrição                      |
|-----------------------|----------------------------|--------------------------------|
| `PING\n`             | `OK PONG\n`                | Teste de conectividade         |
| `STATUS\n`           | `OK STATUS READY\n`        | Status do supervisor           |
| `RESET\n`            | `OK RESET_DONE\n`          | Pulso EN/RST 300 ms           |
| `RESET 500\n`        | `OK RESET_DONE 500\n`      | Pulso customizado (50–3000 ms) |
| `POWER_CYCLE\n`      | `OK POWER_CYCLE_DONE\n`    | Power-cycle 3 s (default)      |
| `POWER_CYCLE 5000\n` | `OK POWER_CYCLE_DONE 5000\n` | Power-cycle customizado (1–10 s) |

Comandos não reconhecidos retornam `ERR UNKNOWN_CMD\n`.

## Integração com o RIFT-V

No YAML do experimento:

```yaml
serial:
  port: COM19               # ESP32 target
  supervisor_port: COM22    # Pico supervisor
```

O framework (`riftv/supervisor.py`) conecta ao Pico automaticamente e o
utiliza como primeira estratégia de recuperação:

1. **Pico supervisor: USB power-cycle** ← este mecanismo
2. DTR/RTS classic toggle
3. USB-JTAG-Serial sequence
4. Serial BREAK
5. Watchdog wait

Recuperações são registradas no `results.jsonl` como:

```json
{"recovery_method": "external_pico_supervisor", "boot_seen": true}
```

## Teste rápido (Python)

```python
from riftv.supervisor import PicoSupervisor

with PicoSupervisor("COM22") as sup:
    print(sup.ping())          # OK PONG
    print(sup.status())        # OK STATUS READY
    print(sup.power_cycle())   # OK POWER_CYCLE_DONE 3000
```

## Identificando as portas COM (Windows)

1. Abra o **Gerenciador de Dispositivos** → Portas (COM & LPT).
2. Desplugue o Pico → anote qual porta desaparece → essa é `supervisor_port`.
3. Desplugue o ESP32 → anote qual porta desaparece → essa é `port`.
