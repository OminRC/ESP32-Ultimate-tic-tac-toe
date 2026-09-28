# نسخه‌ی Arduino IDE

همون منطق بازی و مدل هوش مصنوعی پروژه‌ی ESP-IDF (پوشه‌ی `../main/`)، فقط
تبدیل‌شده به یه اسکچ عادی Arduino (`setup()`/`loop()`، با `Serial` برای
ورودی/خروجی) به‌جای کامپوننت ESP-IDF.

## راه‌اندازی (Arduino IDE)

۱. **نصب پشتیبانی برد ESP32** (اگه قبلاً نصب نشده):
   - از منوی File → Preferences → قسمت "Additional Boards Manager URLs" این
     آدرس رو اضافه کن:
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager → جستجوی "esp32" → نصب پکیج
     **esp32 by Espressif Systems**

۲. **باز کردن اسکچ**: File → Open → فایل `ultimate_ttt/ultimate_ttt.ino` رو
   انتخاب کن (کل پوشه‌ی `ultimate_ttt` باید کنار هم بمونه — Arduino خودش
   همه‌ی فایل‌های `.c`/`.h` کنار `.ino` رو کامپایل می‌کنه، نیازی به
   Makefile/CMake نیست).

۳. **انتخاب برد**: Tools → Board → esp32 → ماژول دقیقت رو انتخاب کن.
   اگه "ESP32S3 Dev Module" برای یه ماژول لخت N8R2 توی لیست نبود، مشکلی
   نیست — گزینه‌ی عمومی ESP32S3 رو انتخاب کن و این‌ها رو تنظیم کن:
   - Tools → Flash Size → 8MB
   - Tools → PSRAM → OPI PSRAM (یا "Enabled"، بسته به نسخه‌ی core)
   - Tools → Partition Scheme → Default (یا "Huge APP" اگه خطای «حجم
     زیاد» گرفتی، که بعیده چون این مدل خیلی کوچیکه)

۴. **انتخاب پورت**: Tools → Port → پورت COM مبدل USB-TTL‌ت رو انتخاب کن
   (همون سیم‌کشی که برای فلش ESP-IDF گفته بودم: 3V3، GND، TXD0، RXD0، و
   موقع ریست، IO0 رو به GND وصل کن تا وارد حالت دانلود بشه).

۵. **آپلود**: دکمه‌ی Upload (فلش) رو بزن. Arduino IDE برای اکثر بردها
   خودش وارد/خارج شدن از حالت دانلود رو مدیریت می‌کنه؛ اگه این کار رو
   نکرد (ماژول لخت، بدون مدار auto-reset)، درست همون لحظه‌ی شروع آپلود،
   دستی IO0 رو به GND بزن و EN رو ریست کن.

۶. **باز کردن Serial Monitor**: Tools → Serial Monitor، نرخ باد (baud
   rate) رو روی **115200** بذار، و line ending رو روی "Newline" تنظیم
   کن. صفحه‌ی بازی چاپ می‌شه؛ حرکتت رو به‌صورت `sub cell` تایپ کن
   (مثلاً `4 4`) و Enter بزن.

## فایل‌ها

- `ultimate_ttt.ino` — نقطه‌ی ورود Arduino (`setup()`/`loop()`)، همون حلقه‌ی
  بازی `../main/main.c` رو پیاده‌سازی می‌کنه ولی با `Serial` به‌جای
  کنسول stdio مخصوص ESP-IDF.
- `game.c` / `game.h` — موتور بازی، کپی دقیق از `../main/`.
- `net_infer.c` / `net_infer.h` — inference شبکه‌ی عصبی، کپی دقیق.
- `model_weights.h` — وزن‌های train‌شده. هروقت checkpoint جدیدی از
  `train/export_weights.py` یا نوت‌بوک Kaggle export گرفتی، **این فایل
  رو جایگزین کن**.
