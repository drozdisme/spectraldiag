# Что было сломано и как это исправлено

## Корневая причина всех трёх падений

Старый подход: C++ компилировался в "голый" `.so` и грузился через `cffi.dlopen()`.
Этот `.so` лежал закоммиченным в репозитории. Весь wheel-тулчейн
(`auditwheel` на Linux, `delocate` на macOS) считает такой файл Python-расширением
собранным под чужую платформу — и падает.

## Решение: настоящий CPython extension module

`src/pymodule.cpp` — теперь это полноценный модуль с `PyInit__core()`.
- `auditwheel`/`delocate` понимают его нативно
- `setuptools` сам кладёт `.so` в wheel с правильным тегом платформы
- `setup.py` стал тривиальным (8 строк), без самодельной логики компиляции
- больше нет зависимости от `cffi`

## ⚠️ ОБЯЗАТЕЛЬНЫЙ РУЧНОЙ ШАГ

В вашем репозитории закоммичен старый `_core.so`. `.gitignore` его НЕ удаляет.
Выполните один раз:

```bash
git rm --cached python/spectraldiag/_core.so
git rm --cached python/spectraldiag/_core.cpython-*.so 2>/dev/null || true
git add .gitignore
git commit -m "Remove committed .so, switch to real CPython extension"
git push
```

Без этого Linux-сборка продолжит падать на auditwheel, потому что
файл всё ещё в git.

## Что изменилось в файлах

- `src/pymodule.cpp` — НОВЫЙ: CPython C-API обёртка
- `setup.py` — упрощён до стандартного Extension
- `pyproject.toml` — убрана зависимость cffi
- `python/spectraldiag/__init__.py` — использует extension module вместо cffi
- `tests/test_core.py` — импорт через публичный API
- `.gitignore` — игнорирует все `*.so`/`*.pyd`
- `.github/workflows/build.yml` — добавлен CIBW_TEST_SKIP для arm64 (нельзя тестировать arm64 wheel на x86_64 раннере)
- удалены неиспользуемые `src/c_api.hpp`, `src/c_exports.cpp`, `src/bindings.cpp`

## Проверено локально

- wheel собирается через `pip wheel` (как в cibuildwheel) ✓
- `.so` внутри wheel с правильным тегом ✓
- wheel ставится в чистый venv и тесты проходят ✓
- sdist пересобирает рабочий wheel с нуля ✓
