using System;

namespace SAI_3.Neural
{
    /// <summary>
    /// Предобработка входных данных перед подачей в нейросеть.
    /// <para>
    /// Содержит утилиты для нормализации нарисованного изображения цифры
    /// к фиксированному формату 5x7 (35 признаков).
    /// </para>
    /// </summary>
    public static class Preprocess
    {
        /// <summary>
        /// Обрезает изображение по ограничивающему прямоугольнику (bounding box)
        /// и масштабирует результат обратно в размер 5x7.
        /// <para>
        /// Используется для центрирования и нормализации цифры, чтобы сеть была
        /// менее чувствительна к сдвигам и различному размеру рисунка.
        /// </para>
        /// </summary>
        /// <param name="x">
        /// Входной вектор длиной 35 (изображение 5x7),
        /// значения ожидаются в диапазоне [0; 1] (обычно 0 или 1).
        /// </param>
        /// <returns>
        /// Новый вектор длиной 35 (5x7), содержащий нормализованное изображение.
        /// Если вход пустой (нет ни одного закрашенного пикселя),
        /// возвращается исходный вектор <paramref name="x"/>.
        /// </returns>
        public static double[] CropAndNormalizeTo5x7(double[] x)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));

            const int W = 5;
            const int H = 7;

            int minR = H, minC = W, maxR = -1, maxC = -1;

            for (int r = 0; r < H; r++)
            {
                for (int c = 0; c < W; c++)
                {
                    // Считаем пиксель "включённым", если значение >= 0.5
                    if (x[r * W + c] < 0.5) continue;

                    if (r < minR) minR = r;
                    if (c < minC) minC = c;
                    if (r > maxR) maxR = r;
                    if (c > maxC) maxC = c;
                }
            }

            // Если не найдено ни одного закрашенного пикселя — возвращаем исходный вектор
            if (maxR < 0)
                return x;

            int cropH = maxR - minR + 1;
            int cropW = maxC - minC + 1;

            var crop = new double[cropW * cropH];

            for (int r = 0; r < cropH; r++)
            {
                for (int c = 0; c < cropW; c++)
                {
                    crop[r * cropW + c] = x[(minR + r) * W + (minC + c)];
                }
            }

            // Масштабирование вырезанного изображения обратно в 5x7
            // Метод: ближайший сосед (nearest neighbor)
            var outImg = new double[W * H];

            for (int r = 0; r < H; r++)
            {
                int srcR = (int)Math.Round((r + 0.5) * cropH / (double)H - 0.5);
                srcR = Math.Clamp(srcR, 0, cropH - 1);

                for (int c = 0; c < W; c++)
                {
                    int srcC = (int)Math.Round((c + 0.5) * cropW / (double)W - 0.5);
                    srcC = Math.Clamp(srcC, 0, cropW - 1);

                    outImg[r * W + c] = crop[srcR * cropW + srcC];
                }
            }

            return outImg;
        }
    }
}