using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SAI_3.Data
{
    /// <summary>
    /// Один пример данных: входной вектор X (например, 35 пикселей 5x7) и метка класса (0..9).
    /// </summary>
    public sealed class Sample
    {
        public double[] X { get; }
        public int Label { get; }

        public Sample(double[] x, int label)
        {
            X = x;
            Label = label;
        }
    }
}
