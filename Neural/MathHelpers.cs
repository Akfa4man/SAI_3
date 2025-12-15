using System;
using System.Collections.Generic;

namespace SAI_3.Neural
{
    /// <summary>
    /// Набор минимальных математических утилит для реализации простого MLP
    /// без сторонних библиотек (активации, softmax, служебные операции).
    /// </summary>
    public static class MathHelpers
    {
        /// <summary>
        /// Функция активации ReLU: <c>max(0, x)</c>.
        /// Обычно применяется на скрытых слоях, чтобы добавить нелинейность.
        /// </summary>
        /// <param name="x">Входное значение (предактивация нейрона).</param>
        /// <returns>0, если <paramref name="x"/> &lt;= 0; иначе возвращает <paramref name="x"/>.</returns>
        public static double ReLU(double x)
        {
            return x > 0.0 ? x : 0.0;
        }

        /// <summary>
        /// Производная ReLU по входу.
        /// <para>
        /// В точке x = 0 производная формально не определена; в практических реализациях
        /// обычно используют 0 или 1. Здесь используется 0 при x &lt;= 0.
        /// </para>
        /// </summary>
        /// <param name="x">Входное значение (предактивация нейрона).</param>
        /// <returns>1 при <paramref name="x"/> &gt; 0; иначе 0.</returns>
        public static double ReLUDerivative(double x)
        {
            return x > 0.0 ? 1.0 : 0.0;
        }

        /// <summary>
        /// Численно стабильный Softmax: преобразует массив логитов в распределение вероятностей.
        /// <para>
        /// Используется на выходном слое при многоклассовой классификации:
        /// на вход подаются "сырые" выходы сети (logits), на выходе — вероятности классов,
        /// сумма которых равна 1.
        /// </para>
        /// <para>
        /// Для стабильности из каждого логита вычитается максимум (лог-сдвиг),
        /// чтобы избежать переполнения экспоненты.
        /// </para>
        /// </summary>
        /// <param name="logits">Массив логитов (предсказаний до нормализации).</param>
        /// <returns>Новый массив вероятностей той же длины, что и <paramref name="logits"/>.</returns>
        public static double[] Softmax(double[] logits)
        {
            double max = logits[0];
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > max) max = logits[i];

            double sum = 0.0;
            var probs = new double[logits.Length];

            for (int i = 0; i < logits.Length; i++)
            {
                double e = Math.Exp(logits[i] - max);
                probs[i] = e;
                sum += e;
            }

            double inv = 1.0 / Math.Max(sum, 1e-12);
            for (int i = 0; i < probs.Length; i++)
                probs[i] *= inv;

            return probs;
        }

        /// <summary>
        /// Возвращает индекс максимального элемента массива.
        /// <para>
        /// Обычно применяется для получения предсказанного класса по вероятностям/логитам.
        /// </para>
        /// </summary>
        /// <param name="values">Массив значений (например, вероятности классов).</param>
        /// <returns>Индекс элемента с максимальным значением.</returns>
        public static int ArgMax(double[] values)
        {
            int best = 0;
            double bestVal = values[0];

            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > bestVal)
                {
                    bestVal = values[i];
                    best = i;
                }
            }

            return best;
        }

        /// <summary>
        /// Перемешивает элементы списка на месте алгоритмом Fisher–Yates.
        /// Используется для случайного порядка обучения.
        /// </summary>
        /// <typeparam name="T">Тип элементов списка.</typeparam>
        /// <param name="list">Список, который нужно перемешать (изменяется на месте).</param>
        /// <param name="rng">Источник случайных чисел (seed влияет на воспроизводимость).</param>
        public static void Shuffle<T>(IList<T> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        /// <summary>
        /// Генерирует нормально распределённое случайное число N(0,1)
        /// методом Бокса–Мюллера.
        /// <para>
        /// Обычно применяется для инициализации весов (малые случайные значения).
        /// </para>
        /// </summary>
        /// <param name="rng">Источник случайных чисел.</param>
        /// <returns>Случайное число с нормальным распределением (среднее 0, дисперсия 1).</returns>
        public static double NextGaussian(Random rng)
        {
            // u1 не должен быть 0, иначе log(0) -> -Infinity.
            double u1 = Math.Max(1e-12, rng.NextDouble());
            double u2 = rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}