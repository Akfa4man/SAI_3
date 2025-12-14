using SAI_3.Neural;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace SAI_3
{
    public static class ModelIO
    {
        private sealed class ModelDto
        {
            public int Input { get; set; }
            public int Hidden { get; set; }
            public int Output { get; set; }

            public double[][] W1 { get; set; } = Array.Empty<double[]>();
            public double[] B1 { get; set; } = Array.Empty<double>();
            public double[][] W2 { get; set; } = Array.Empty<double[]>();
            public double[] B2 { get; set; } = Array.Empty<double>();
        }

        public static void Save(Mlp net, string path)
        {
            if (net == null) throw new ArgumentNullException(nameof(net));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Bad path", nameof(path));

            var (w1, b1, w2, b2) = net.GetParametersCopy();

            var dto = new ModelDto
            {
                Input = net.InputSize,
                Hidden = net.HiddenSize,
                Output = net.OutputSize,
                W1 = w1,
                B1 = b1,
                W2 = w2,
                B2 = b2
            };

            var opt = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(dto, opt);

            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllText(path, json);
        }

        public static Mlp Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Model file not found", path);

            var json = File.ReadAllText(path);
            var dto = JsonSerializer.Deserialize<ModelDto>(json)
              ?? throw new InvalidOperationException("Bad model json");

            var net = new Mlp(dto.Input, dto.Hidden, dto.Output, seed: 1);
            net.SetParameters(dto.W1, dto.B1, dto.W2, dto.B2);
            return net;
        }
    }
}
