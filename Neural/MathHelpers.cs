using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SAI_3.Neural
{
    /// <summary>
    /// Небольшой набор вспомогательных математических функций,
    /// необходимых для простого MLP (без сторонних библиотек).
    /// </summary>
    public static class MathHelpers
    {
        /// <summary>
        /// ReLU-активация.
        /// </summary>
        public static double ReLU(double x)
        {
            return x > 0.0 ? x : 0.0;
        }

        /// <summary>
        /// Производная ReLU.
        /// </summary>
        public static double ReLUDerivative(double x)
        {
            return x > 0.0 ? 1.0 : 0.0;
        }

        /// <summary>
        /// Стабильный Softmax.
        /// На вход — массив логитов, на выход — массив вероятностей.
        /// </summary>
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
        /// Индекс максимального элемента массива.
        /// </summary>
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
        /// Перемешивание списка (алгоритм Фишера-Йейтса).
        /// </summary>
        public static void Shuffle<T>(IList<T> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        /// <summary>
        /// Генерация нормально распределённого случайного числа
        /// (метод Бокса–Мюллера).
        /// </summary>
        public static double NextGaussian(Random rng)
        {
            double u1 = Math.Max(1e-12, rng.NextDouble());
            double u2 = rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
