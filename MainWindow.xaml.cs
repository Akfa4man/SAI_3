using SAI_3.Data;
using SAI_3.Neural;
using System.Globalization;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace SAI_3
{
    public partial class MainWindow : Window
    {
        private readonly bool[] _pixels = new bool[35];

        private Mlp? _net;
        private List<Sample>? _train;
        private List<Sample>? _test;

        private CancellationTokenSource? _cts;

        private readonly string _modelPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model.json");
        private readonly string _samplesPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "samples.json");

        private bool IsTraining => _cts != null;
        private bool HasModel => _net != null;
        private bool HasAnyPixel => Array.Exists(_pixels, p => p);



        public MainWindow()
        {
            InitializeComponent();
            UpdatePixelVisuals();
            UpdateUiState();

            try
            {
                if (System.IO.File.Exists(_modelPath))
                {
                    _net = ModelIO.Load(_modelPath);
                    StatusText.Text = "Загружена модель из model.json";
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = "Не удалось загрузить model.json: " + ex.Message;
            }
        }

        private void UpdateUiState()
        {
            TrainButton.IsEnabled = !IsTraining;
            StopButton.IsEnabled = IsTraining;

            PredictButton.IsEnabled = HasModel && HasAnyPixel && !IsTraining;

            RememberPanel.IsEnabled = HasAnyPixel && !IsTraining;

            SaveModelButton.IsEnabled = HasModel && !IsTraining;
            LoadModelButton.IsEnabled = !IsTraining && System.IO.File.Exists(_modelPath);

            ClearButton.IsEnabled = !IsTraining;

            EpochsBox.IsEnabled = !IsTraining;
            BatchSizeBox.IsEnabled = !IsTraining;
            LearningRateBox.IsEnabled = !IsTraining;
            NoiseBox.IsEnabled = !IsTraining;
        }

        private void Pixel_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not ToggleButton tb) return;
            if (!int.TryParse(tb.Tag?.ToString(), out int idx)) return;
            if (idx < 0 || idx >= _pixels.Length) return;

            _pixels[idx] = tb.IsChecked == true;
            tb.Background = _pixels[idx] ? Brushes.Black : Brushes.White;

            UpdateUiState();
        }

        private void ClearButton_Click(object sender, RoutedEventArgs e)
        {
            Array.Clear(_pixels, 0, _pixels.Length);
            PredText.Text = "";
            ProbsText.Text = "";
            UpdatePixelVisuals();

            UpdateUiState();
        }

        private void UpdatePixelVisuals()
        {
            foreach (var tb in FindVisualChildren<ToggleButton>(this))
            {
                if (!int.TryParse(tb.Tag?.ToString(), out int idx)) continue;
                if (idx < 0 || idx >= _pixels.Length) continue;

                tb.IsChecked = _pixels[idx];
                tb.Background = _pixels[idx] ? Brushes.Black : Brushes.White;
            }
        }

        private static IEnumerable<T> FindVisualChildren<T>(DependencyObject depObj) where T : DependencyObject
        {
            if (depObj == null) yield break;

            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(depObj); i++)
            {
                var child = VisualTreeHelper.GetChild(depObj, i);
                if (child is T t) yield return t;

                foreach (var childOfChild in FindVisualChildren<T>(child))
                    yield return childOfChild;
            }
        }

        private void PredictButton_Click(object sender, RoutedEventArgs e)
        {
            if (IsTraining) return;


            if (_net == null)
            {
                MessageBox.Show("Сначала обучите сеть (Train).", "Нет модели", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            double[] x = _pixels.Select(p => p ? 1.0 : 0.0).ToArray();

            int pred = _net.Predict(x);
            var probs = _net.PredictProba(x);

            PredText.Text = pred.ToString(CultureInfo.InvariantCulture);

            ProbsText.Text = string.Join("  ",
              probs.Select((p, i) => $"{i}:{p:0.000}")
            );
        }

        private void SaveModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (_net == null)
            {
                MessageBox.Show("Нет модели для сохранения.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            try
            {
                ModelIO.Save(_net, _modelPath);
                StatusText.Text = "Модель сохранена в model.json";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Save error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            UpdateUiState();
        }

        private void LoadModelButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                _net = ModelIO.Load(_modelPath);
                StatusText.Text = "Модель загружена из model.json";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Load error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            UpdateUiState();
        }

        private void RememberDigit_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not System.Windows.Controls.Button btn) return;
            if (!int.TryParse(btn.Content?.ToString(), out int label)) return;

            double[] x = _pixels.Select(p => p ? 1.0 : 0.0).ToArray();

            try
            {
                SamplesIO.Append(_samplesPath, new Sample(x, label));
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Samples error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            _net ??= new Mlp(inputSize: 35, hiddenSize: 128, outputSize: 10, seed: 42);

            try
            {
                var all = SamplesIO.Load(_samplesPath);
                var rng = new Random(999);
                int epochs = 25;
                int batch = Math.Min(16, Math.Max(1, all.Count));
                double lr = 0.03;

                for (int ep = 0; ep < epochs; ep++)
                    _net.TrainEpoch(all, batch, lr, rng);

                ModelIO.Save(_net, _modelPath);

                StatusText.Text = $"Пример сохранён (label={label}). Дообучено на {all.Count} примерах. Модель обновлена.";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Fine-tune error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            UpdateUiState();
        }

        private async void TrainButton_Click(object sender, RoutedEventArgs e)
        {
            if (_cts != null)
            {
                MessageBox.Show("Обучение уже запущено.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            if (!TryReadTrainingParams(out int epochs, out int batchSize, out double lr, out double noise))
                return;

            _train = DatasetGenerator.Generate(perClass: 400, noiseFlipProb: noise, seed: 123, shuffle: true);
            _test = DatasetGenerator.Generate(perClass: 60, noiseFlipProb: noise, seed: 456, shuffle: true);

            _net = new Mlp(inputSize: 35, hiddenSize: 64, outputSize: 10, seed: 42);

            TrainButton.IsEnabled = false;
            StopButton.IsEnabled = true;
            TrainProgress.Value = 0;
            StatusText.Text = "Старт обучения...";

            _cts = new CancellationTokenSource();
            var token = _cts.Token;

            UpdateUiState();

            try
            {
                await Task.Run(() => TrainLoop(_net, _train, _test, epochs, batchSize, lr, token), token);
            }
            catch (OperationCanceledException)
            {
                StatusText.Text = "Обучение остановлено пользователем.";
            }
            catch (Exception ex)
            {
                StatusText.Text = "Ошибка обучения: " + ex.Message;
            }
            finally
            {
                _cts?.Dispose();
                _cts = null;

                TrainButton.IsEnabled = true;
                StopButton.IsEnabled = false;

                UpdateUiState();
            }
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            _cts?.Cancel();
            UpdateUiState();
        }

        private void TrainLoop(Mlp net, List<Sample> train, List<Sample> test,
          int epochs, int batchSize, double lr, CancellationToken token)
        {
            var rng = new Random(777);

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                if (token.IsCancellationRequested)
                    break;

                double loss = net.TrainEpoch(train, batchSize, lr, rng);
                double trainAcc = net.EvaluateAccuracy(train);
                double testAcc = net.EvaluateAccuracy(test);

                int progress = (int)Math.Round(epoch * 100.0 / epochs);

                Dispatcher.Invoke(() =>
                {
                    TrainProgress.Value = progress;
                    StatusText.Text =
                      $"Epoch {epoch}/{epochs}  |  loss={loss:0.0000}  |  trainAcc={trainAcc * 100:0.00}%  |  testAcc={testAcc * 100:0.00}%";
                });
            }

            Dispatcher.Invoke(() =>
            {
                StatusText.Text += "\nГотово. Можно рисовать цифру и нажимать Predict.";
            });
        }

        private bool TryReadTrainingParams(out int epochs, out int batchSize, out double lr, out double noise)
        {
            epochs = 0; batchSize = 0; lr = 0; noise = 0;

            if (!int.TryParse(EpochsBox.Text, out epochs) || epochs <= 0)
            {
                MessageBox.Show("Epochs должно быть целым > 0", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            if (!int.TryParse(BatchSizeBox.Text, out batchSize) || batchSize <= 0)
            {
                MessageBox.Show("BatchSize должно быть целым > 0", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            if (!double.TryParse(LearningRateBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out lr) || lr <= 0)
            {
                MessageBox.Show("LearningRate должно быть числом > 0 (используй точку, например 0.05)", "Ошибка",
                  MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            if (!double.TryParse(NoiseBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out noise) || noise < 0 || noise > 1)
            {
                MessageBox.Show("NoiseFlipProb должно быть числом в диапазоне [0..1]", "Ошибка",
                  MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            return true;
        }
    }
}