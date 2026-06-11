# kronos_lab 

kronos_lab/
├── config.py                <- ВСЕ настройки: монеты, комиссия, параметры, пути
├── foundation/              <- чистая логика
│   ├── ensemble.py          контракт данных + save/load + синтетика
│   ├── generation.py        форк inference, сбор ансамбля (нужен gpu)
│   ├── analytics.py         форма распределения: skew, хвосты, VaR, конус
│   ├── journal.py           калибровка + бумажная торговля, независимый срез
│   └── bybit_data.py        весь bybit: контекст + исход
├── cli/                     <- скрипты запуска
│   ├── run_cycle.py         главный цикл по всем монетам
│   ├── fetch_actual.py      закрыть созревшие исходы
│   ├── evaluate_journal.py  отчёт калибровки и P&L
│   ├── record_signal.py     ручная запись ансамбля в журнал
│   └── compare_ensembles.py сравнить два ансамбля
├── data/
│   ├── ensembles/           .npz ансамбли
│   └── journal.csv          единый журнал всех монет (создаётся сам)
└── README.md


~/aProject/trading/
├── kronos_model/      <- библиотека модели (уже есть)
└── kronos_lab/        <- наш стенд

config.py сам найдёт kronos_model по относительному пути

старый разрозненный мусор из корня можно архивировать:
```bash
cd ~/aProject/trading
mkdir -p _old
mv smoke_generate.py compare_ensembles.py record_signal.py fetch_actual.py run_cycle.py _old/ 2>/dev/null
mv logs/journal.csv _old/journal_old_smoke.csv 2>/dev/null
(старый журнал был на 27 колонок и модели small, новый стенд пишет свой с нуля)

## запуск

всё запускать из папки kronos_lab:
```bash
cd ~/aProject/trading/kronos_lab

# один проход по всем монетам (генерация + запись + закрытие созревших + отчёт)
python cli/run_cycle.py

# только отчёт по накопленному
python cli/evaluate_journal.py

# только закрыть созревшие исходы
python cli/fetch_actual.py

если CUDA out of memory на base - уменьшить CHUNK в config.py (8 -> 4)

## cron (когда ручной прогон отработает чисто)

```bash
crontab -e
5 * * * * cd ~/aProject/trading/kronos_lab && /home/grant/aProject/trading/venv_kronos/bin/python cli/run_cycle.py >> data/cycle.log 2>&1
:05 каждого часа - чтобы часовая свеча успела закрыться

## как читать отчёт

- калибровка считается по ВСЕМ строкам (часовым), так плавнее
- P&L показывается дважды: по всем строкам (перекрытые, осторожно) и по НЕЗАВИСИМОМУ суточному срезу
- доверять по доходности только независимому срезу - перекрытые часовые сделки раздувают цифру
- разбивка по монетам показывает, где модель калибрована лучше

## что НЕ делает

денег не трогает. это сбор статистики. реальная торговля - после недели данных и только если в независимом срезе виден край
