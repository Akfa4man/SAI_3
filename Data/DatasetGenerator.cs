using System;
using System.Collections.Generic;

namespace SAI_3.Data
{
    /// <summary>
    /// Генератор синтетического набора данных для распознавания цифр 0..9.
    /// Строит примеры на основе эталонных 5x7 шаблонов (DigitTemplates) и,
    /// при необходимости, добавляет "шум" путём инверсии отдельных пикселей.
    /// </summary>
    public static class DatasetGenerator
    {
        /// <summary>
        /// Создаёт список обучающих примеров <see cref="Sample"/> для цифр 0..9.
        /// Каждый пример строится из 5x7 эталонного вектора конкретной цифры и может
        /// быть искажён шумом: каждый пиксель (бит) независимо инвертируется с вероятностью <paramref name="noiseFlipProb"/>.
        /// </summary>
        /// <param name="perClass">Количество примеров, генерируемых для каждой цифры (класса).</param>
        /// <param name="noiseFlipProb">
        /// Вероятность инверсии каждого пикселя (0..1).
        /// 0 — без шума, 1 — инверсия всех пикселей.
        /// </param>
        /// <param name="seed">
        /// Seed для генератора случайных чисел. Позволяет получать воспроизводимый датасет при одинаковых параметрах.
        /// </param>
        /// <param name="shuffle">
        /// Если <c>true</c>, итоговый список перемешивается (Fisher–Yates), чтобы примеры разных классов были вперемешку.
        /// </param>
        /// <returns>
        /// Список примеров <see cref="Sample"/>, где вход — вектор признаков (пикселей), а метка — цифра 0..9.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Выбрасывается, если <paramref name="perClass"/> &lt;= 0 или <paramref name="noiseFlipProb"/> вне диапазона [0; 1].
        /// </exception>
        public static List<Sample> Generate(
            int perClass,
            double noiseFlipProb,
            int seed = 123,
            bool shuffle = true)
        {
            if (perClass <= 0) throw new ArgumentOutOfRangeException(nameof(perClass));
            if (noiseFlipProb < 0 || noiseFlipProb > 1) throw new ArgumentOutOfRangeException(nameof(noiseFlipProb));

            var rng = new Random(seed);

            var data = new List<Sample>(perClass * 10);

            for (int digit = 0; digit <= 9; digit++)
            {
                double[] baseVec = DigitTemplates.GetVector(digit);

                for (int i = 0; i < perClass; i++)
                {
                    var x = (double[])baseVec.Clone();

                    if (noiseFlipProb > 0)
                    {
                        for (int p = 0; p < x.Length; p++)
                        {
                            if (rng.NextDouble() < noiseFlipProb)
                                x[p] = 1.0 - x[p];
                        }
                    }

                    data.Add(new Sample(x, digit));
                }
            }

            if (shuffle)
                ShuffleInPlace(data, rng);

            return data;
        }

        /// <summary>
        /// Перемешивает элементы списка на месте алгоритмом Fisher–Yates.
        /// Использует переданный генератор случайных чисел для воспроизводимости.
        /// </summary>
        /// <typeparam name="T">Тип элементов списка.</typeparam>
        /// <param name="list">Список, который нужно перемешать (изменяется на месте).</param>
        /// <param name="rng">Источник случайных чисел.</param>
        private static void ShuffleInPlace<T>(IList<T> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }
    }
}