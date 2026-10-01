# Ultimate Tic-Tac-Toe AI on ESP32-S3

[English](#english) | [فارسی](#فارسی)

An AlphaZero-lite policy+value network for [Ultimate Tic-Tac-Toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe),
trained via self-play MCTS (pure NumPy/PyTorch, no TFLite Micro), exported as
plain C float arrays, and run as greedy single-forward-pass inference on an
**ESP32-S3**. No external sensors or display modules needed — play against
it over the Serial monitor.

```
[self-play + MCTS]  -->  [tiny MLP: 126 -> 96 -> 64 -> {81, 1}]  -->  [C float arrays]  -->  [ESP32-S3 inference]
      (PC / Kaggle GPU)              ~24K params, ~93KB                                        Serial-monitor game
```

> **Target hardware**: this project was built and tuned specifically for the
> **ESP32-S3-N8R2** module (8MB flash / 512KB SRAM / 2MB PSRAM). Every design
> choice — model size (~93KB), skipping TFLite Micro, skipping on-device
> MCTS — was made to comfortably fit *this* chip's memory budget, not as a
> generic "runs on any ESP32" claim. It should also run unmodified on other
> ESP32-S3 variants with at least this much flash/RAM (e.g. N16R8), but
> hasn't been tested on the plain ESP32, ESP32-C3, or other non-S3 chips.

---

## English

### Repository layout

```
train/      Training pipeline (Python) -- self-play MCTS, network, export to C
main/       ESP-IDF project: play over the Serial monitor
arduino/    Arduino IDE versions: ultimate_ttt/ (Serial monitor), ultimate_ttt_wifi/ (phone browser over WiFi)
wifi/       ESP-IDF project: board hosts its own WiFi + a web UI -- play from a phone browser, no cable needed after flashing (see wifi/README.md)
```

### Quick start

#### 1. Train a model

Locally (NumPy, CPU, slow but dependency-free):
```bash
cd train
pip install -r requirements.txt
python train.py --iterations 25 --games-per-iter 25 --sims 50 --out policy_params.npz
python eval_vs_random.py policy_params.npz --games 100
```

Or on [Kaggle](https://www.kaggle.com) with a free GPU (much faster; see
[`train/KAGGLE.md`](train/KAGGLE.md) for the full walkthrough, and
[`train/ultimate_ttt_kaggle.ipynb`](train/ultimate_ttt_kaggle.ipynb) for a
ready-to-import notebook). `train_kaggle_mp.py` runs self-play across
multiple CPU-core worker processes (the model is too small for the GPU
itself to be the bottleneck) and includes early stopping against a
random-move baseline, so you don't have to guess how many iterations you
need.

#### 2. Export the trained weights to C

```bash
python export_weights.py policy_params.npz --out ../main/model_weights.h
```

The model is ~93KB as float32 — small enough that no INT8 quantization is
needed; it just lives in flash as a `const` array.

#### 3. Build and flash

**ESP-IDF:**
```bash
cd ..
idf.py set-target esp32s3
idf.py -p <PORT> build flash monitor
```

**Arduino IDE:** open [`arduino/ultimate_ttt/ultimate_ttt.ino`](arduino/ultimate_ttt/ultimate_ttt.ino) — see
[`arduino/README.md`](arduino/README.md) for board-manager setup, board
settings (8MB flash, PSRAM enabled), and wiring notes for a bare module
(no onboard USB-UART).

#### 4. Play

The board prints as text over the Serial monitor (115200 baud). You are
`O`, the AI is `X`. Enter moves as `sub cell` (both 0-8), e.g.:

```
> 4 4
```

`sub` = which of the 9 mini-boards, `cell` = which of its 9 cells. The
active-sub-board rule is enforced (the cell you play in picks which
mini-board your opponent must play in next — standard Ultimate
Tic-Tac-Toe rules).

### Design notes

- **Why no TFLite Micro**: the network is tiny (126→96→64→{81,1}, ~24K
  params), so hand-rolled float matmuls in [`main/net_infer.c`](main/net_infer.c)
  are simpler than pulling in the whole TFLite Micro component.
- **Why no on-device search**: MCTS at inference time would need many NN
  forward passes per move; a single greedy forward pass is enough for a
  reasonable opponent and keeps the firmware trivial. Porting
  [`train/mcts.py`](train/mcts.py)'s `run_mcts` to C and calling `net_infer`
  inside it would give a stronger on-device AI (memory cost stays tiny,
  just slower per move) if you want to take it further.
- **Why plain NumPy/PyTorch instead of TensorFlow**: keeps the training
  pipeline dependency-light and avoids TF/protobuf version conflicts.
- **Memory budget**: model weights ~93KB (flash, not SRAM), inference
  scratch buffers well under 1KB SRAM, game state struct ~100 bytes —
  comfortably fits within 512KB SRAM.

### License

MIT — see [LICENSE](LICENSE).

---

## فارسی

یه شبکه‌ی policy+value به سبک AlphaZero-lite برای بازی
[Ultimate Tic-Tac-Toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)،
که با self-play MCTS آموزش دیده (فقط NumPy/PyTorch، بدون TFLite Micro)،
به‌صورت آرایه‌های float در C اکسپورت شده، و روی **ESP32-S3** به‌صورت یه
forward-pass ساده (greedy) اجرا می‌شه. هیچ سنسور یا ماژول نمایشگر
خارجی لازم نیست — از طریق Serial Monitor باهاش بازی کن.

```
[self-play + MCTS]  -->  [MLP کوچیک: ۱۲۶ -> ۹۶ -> ۶۴ -> {۸۱، ۱}]  -->  [آرایه‌های float در C]  -->  [inference روی ESP32-S3]
      (PC یا GPU رایگان Kaggle)         ~۲۴هزار پارامتر، ~۹۳KB                                        بازی از طریق Serial
```

> **سخت‌افزار هدف**: این پروژه دقیقاً برای ماژول **ESP32-S3-N8R2**
> (۸ مگابایت فلش / ۵۱۲ کیلوبایت SRAM / ۲ مگابایت PSRAM) ساخته و تنظیم
> شده. هر تصمیم طراحی — حجم مدل (~۹۳KB)، رد کردن TFLite Micro، رد کردن
> جستجوی MCTS روی خود برد — دقیقاً برای جا‌شدن راحت داخل بودجه‌ی حافظه‌ی
> **همین چیپ** گرفته شده، نه یه ادعای کلی «روی هر ESP32 اجرا می‌شه».
> روی سایر نسخه‌های ESP32-S3 با حداقل همین مقدار فلش/RAM (مثل N16R8) هم
> باید بدون تغییر کار کنه، ولی روی ESP32 معمولی، ESP32-C3، یا چیپ‌های
> غیر-S3 تست نشده.

### ساختار مخزن

```
train/      پایپ‌لاین ترینینگ (پایتون) -- self-play MCTS، شبکه، export به C
main/       پروژه‌ی ESP-IDF: بازی از طریق Serial Monitor
arduino/    نسخه‌های Arduino IDE: ultimate_ttt/ (Serial Monitor)، ultimate_ttt_wifi/ (مرورگر گوشی از طریق WiFi)
wifi/       پروژه‌ی ESP-IDF: برد خودش یه WiFi و وب‌سرور بالا میاره -- بازی از طریق مرورگر گوشی، بدون نیاز به کابل بعد از فلش (جزئیات در wifi/README.md)
```

### شروع سریع

#### ۱. ترینینگ مدل

به‌صورت محلی (NumPy، روی CPU، کند ولی بدون وابستگی سنگین):
```bash
cd train
pip install -r requirements.txt
python train.py --iterations 25 --games-per-iter 25 --sims 50 --out policy_params.npz
python eval_vs_random.py policy_params.npz --games 100
```

یا روی [Kaggle](https://www.kaggle.com) با GPU رایگان (خیلی سریع‌تر؛
راهنمای کامل در [`train/KAGGLE.md`](train/KAGGLE.md)، و یه نوت‌بوک
آماده برای Import در [`train/ultimate_ttt_kaggle.ipynb`](train/ultimate_ttt_kaggle.ipynb)).
اسکریپت `train_kaggle_mp.py` self-play رو بین چند پردازه‌ی موازی روی
هسته‌های CPU پخش می‌کنه (چون مدل اونقدر کوچیکه که GPU خودش تنگنا
نمی‌شه) و یه early stopping در برابر حریف تصادفی هم داره، پس لازم
نیست حدس بزنی چند iteration کافیه.

#### ۲. اکسپورت وزن‌های train‌شده به C

```bash
python export_weights.py policy_params.npz --out ../main/model_weights.h
```

مدل حدود ۹۳ کیلوبایت (float32) هست — اونقدر کوچیکه که نیازی به
کوانتایز INT8 نداره؛ فقط به‌صورت یه آرایه‌ی `const` روی فلش می‌شینه.

#### ۳. Build و Flash

**ESP-IDF:**
```bash
cd ..
idf.py set-target esp32s3
idf.py -p <PORT> build flash monitor
```

**Arduino IDE:** فایل
[`arduino/ultimate_ttt/ultimate_ttt.ino`](arduino/ultimate_ttt/ultimate_ttt.ino)
رو باز کن — راهنمای کامل نصب Board Manager، تنظیمات برد (فلش ۸MB،
PSRAM فعال) و سیم‌کشی برای ماژول لخت (بدون USB-UART روی‌برد) در
[`arduino/README.md`](arduino/README.md) هست.

#### ۴. بازی

صفحه‌ی بازی به‌صورت متن روی Serial Monitor (با baud rate ۱۱۵۲۰۰) چاپ
می‌شه. تو `O` هستی، هوش مصنوعی `X` هست. حرکتت رو به‌صورت `sub cell`
(هر دو بین ۰ تا ۸) وارد کن، مثلاً:

```
> 4 4
```

`sub` = کدوم‌یک از ۹ زیرصفحه، `cell` = کدوم‌یک از ۹ خونه‌ی داخل اون
زیرصفحه. قانون «زیرصفحه‌ی فعال» رعایت می‌شه (خونه‌ای که توش بازی
می‌کنی مشخص می‌کنه حریفت باید تو کدوم زیرصفحه حرکت بعدیش رو بزنه —
قانون استاندارد Ultimate Tic-Tac-Toe).

### نکات طراحی

- **چرا بدون TFLite Micro**: شبکه خیلی کوچیکه (۱۲۶→۹۶→۶۴→{۸۱،۱}،
  حدود ۲۴هزار پارامتر)، برای همین matmul دستی با float در
  [`main/net_infer.c`](main/net_infer.c) از آوردن کل کامپوننت TFLite
  Micro ساده‌تره.
- **چرا بدون جستجوی on-device**: MCTS در لحظه‌ی inference به چندین
  forward pass در هر حرکت نیاز داره؛ یه forward pass ساده (greedy)
  برای یه حریف معقول کافیه و فریمور رو ساده نگه می‌داره. اگه بخوای
  فراتر بری، می‌تونی `run_mcts` از [`train/mcts.py`](train/mcts.py)
  رو به C پورت کنی و `net_infer` رو توش صدا بزنی (هزینه‌ی حافظه هنوز
  کمه، فقط هر حرکت کندتر می‌شه).
- **چرا NumPy/PyTorch خالص به‌جای TensorFlow**: پایپ‌لاین ترینینگ رو
  سبک نگه می‌داره و از تداخل نسخه‌ی TF/protobuf جلوگیری می‌کنه.
- **بودجه‌ی حافظه**: وزن‌های مدل ~۹۳KB (روی فلش، نه SRAM)، بافرهای
  inference خیلی کمتر از ۱KB SRAM، ساختار وضعیت بازی ~۱۰۰ بایت — کاملاً
  داخل ظرفیت ۵۱۲KB SRAM جا می‌شه.

### لایسنس

MIT — به فایل [LICENSE](LICENSE) نگاه کن.
