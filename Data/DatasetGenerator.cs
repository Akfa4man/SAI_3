using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SAI_3.Data
{
    public static class DatasetGenerator
    {
        /// <summary>
        /// Генерирует датасет на основе 5x7-шаблонов цифр (0..9) с добавлением шума.
        /// Шум: с вероятностью noiseFlipProb каждый пиксель инвертируется (0->1, 1->0).
        /// </summary>
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
