.. image:: /docs/ru/logo_ru.png
   :width: 500px
   :align: center
   :alt: Expert Logo in Russian

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |py_9|
   * - license
     - | |license|
   * - languages
     - | |eng| |rus|
.. end-badges

**Эксперт** - это open-source технология, которая предназначена для оценки состоятельности экспертного мнения на основе динамического интеллектуального анализа видеоконтента.

Особенности технологии
==========================================================

.. image:: /docs/ru/diagram_ru.png
    :width: 700px
    :align: center
    :alt: Expert Diagram in Russian

Область применения
==========================================================

Требования
==========================================================

- Python ~=3.9 (python3.9-full, python3.9-dev)
- pip >=22.0 или PDM ~=2.4.8

Установка
==========================================================

Эксперт может быть установлен с помощью ``pip``:

.. code-block:: bash

    $ pip install "expert[all] @ git+https://github.com/expertspec/expert.git"

или с помощью ``pdm``:

.. code-block:: bash

    $ pdm add "expert[all] @ git+https://github.com/expertspec/expert.git"

Запись ``expert[all]`` означает, что будут установлены зависимости из группы ``all``.
Если вы хотите установить зависимости только из группы определенного модуля библиотеки,
то впишите вместо ``all`` название необходимого модуля.
Установка без указания группы зависимостей приведет к установке
библиотеки без зависимостей

Как использовать
==========================================================

Development
==========================================================

Склонировать репозиторий:

.. code-block:: bash

    $ git clone https://github.com/expertspec/expert.git

Установить все зависимости из ``pdm.lock`` файла:

.. code-block:: bash

    $ pdm install -G all

или опциональные зависимости для каждого отдельного модуля библиотеки (см. ``pyproject.toml``):

.. code-block:: bash

    $ pdm install -G <group>

Запустить прекоммитные хуки:

Для обновления версии зависимости (пакета) вам необходимо изменить версию в ``pyproject.toml`` и после выполнения:

.. code-block:: bash

    $ pdm update -G <group> <package>


.. code-block:: bash

    $ pre-commit run (все хуки, только для закоммиченых изменений)
    $ pre-commit run --all-files (все хуки для любых изменений)
    $ pre-commit run <hook_name> (определенный хук)

Примеры
==========================================================

Документация
==========================================================

Публикации об Эксперт
==========================================================

.. [1] Sinko M.V., Medvedev A.A., Smirnov I.Z., Laushkina A.A., Kadnova A., Basov O.O. Method
       of constructing and identifying predictive models of human behavior based on information
       models of non-verbal signals // Procedia Computer Science - 2022, Vol. 212, pp. 171-180

.. [2] Laushkina A., Smirnov I., Medvedev A., Laptev A., Sinko M. Detecting incongruity in the
       expression of emotions in short videos based on a multimodal approach // Cybernetics and
       physics - 2022, Vol. 11, No. 4, pp. 210–216

Благодарности
==========================================================

Контакты
==========================================================

Цитирование
==========================================================

.. |eng| image:: https://img.shields.io/badge/lang-en-red.svg
   :alt: Documentation in English
   :target: /README.rst

.. |rus| image:: https://img.shields.io/badge/lang-ru-deepgreen.svg
   :alt: Documentation in Russian
   :target: /README_ru.rst

.. |py_8| image:: https://img.shields.io/badge/python_3.8-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.8-passing-success

.. |py_9| image:: https://img.shields.io/badge/python_3.9-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.9-passing-success

.. |py_10| image:: https://img.shields.io/badge/python_3.10-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.10-passing-success

.. |license| image:: https://img.shields.io/github/license/expertspec/expert
   :alt: Supported License
   :target: https://github.com/expertspec/expert/blob/master/LICENSE.md
