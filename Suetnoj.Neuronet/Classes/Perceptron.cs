using Suetnoj.Neuronet.Interfaces;
using Suetnoj.Neuronet.Structs;

namespace Suetnoj.Neuronet.Classes
{
    public class Perceptron : INeuronet
    {
        public double[][] Neurons { get; }
        public double[][][] Synapces { get; }
        public IActivator Activator { get; }

        public Perceptron(int[] signature, IActivator activator)
        {
            Activator = activator;
            var random = new Random(DateTime.UtcNow.Millisecond);
            Neurons = new double[signature.Length][];

            for (var l = 0; l < Neurons.Length; l++)
            {
                Neurons[l] = new double[signature[l]];
            }

            Synapces = new double[signature.Length - 1][][];

            for (var l = Synapces.Length - 1; l >= 0; l--)
            {
                Synapces[l] = new double[Neurons[l].Length][];
                for (var n = 0; n < Synapces[l].Length; n++)
                {
                    Synapces[l][n] = new double[Neurons[l + 1].Length];
                    for (var s = 0; s < Synapces[l][n].Length; s++)
                        Synapces[l][n][s] = (double)random.Next(-1, 1) / 10;
                }
            }
        }

        public Task<double[]> Calculate(double[] input)
        {
            var task = Task.Run(() =>
            {
                Neurons[0] = new List<double>(input.AsEnumerable()).ToArray();
                for (var l = 1; l < Neurons.Length; l++)
                for (var n = 0; n < Neurons[l].Length; n++)
                {
                    for (var i = 0; i < Neurons[l - 1].Length; i++)
                        Neurons[l][n] += Neurons[l - 1][i] * Synapces[l - 1][i][n];
                    Neurons[l][n] = Activator.Function(Neurons[l][n]);
                }

                return Neurons[^1];
            });
            return task;
        }

        private Task<double> BackPropagation(double[] target, double factor)
        {
            var task = Task.Run(() =>
            {
                var delta = new double[Neurons.Length][];
                for (var l = 0; l < delta.Length; l++)
                {
                    delta[l] = new double[Neurons[l].Length];
                    for (var n = 0; n < delta[l].Length; n++)
                        delta[l][n] = 0;
                }

                for (var n = 0; n < Neurons[^1].Length; n++)
                {
                    delta[^1][n] = (target[n] - Neurons[^1][n]) * Activator.Derivative(Neurons[^1][n]);
                }

                for (var l = Neurons.Length - 2; l > 0; l--)
                {
                    for (var n = 0; n < Neurons[l].Length; n++)
                    {
                        for (var i = 0; i < Neurons[l + 1].Length; i++)
                        {
                            delta[l][n] += delta[l + 1][i] * Synapces[l][n][i];
                        }

                        delta[l][n] *= Activator.Derivative(Neurons[l][n]);
                    }
                }

                for (var l = 0; l < Neurons.Length - 1; l++)
                {
                    for (var n = 0; n < Neurons[l].Length; n++)
                    {
                        for (var i = 0; i < Neurons[l + 1].Length; i++)
                        {
                            Synapces[l][n][i] += factor * delta[l + 1][i] * Neurons[l][n];
                        }
                    }
                }

                return target.Select((x, i) => Math.Abs(x - Neurons[^1][i])).Prepend(0).Max();
            });
            return task;
        }

        public Task<double> Teach(double[][] inputs, double[][] outputs, TrainingConfiguration configuration,
            CancellationToken cancellationToken)
        {
            var task = Task.Run(async () =>
            {
                double error = 0;
                do
                {
                    for (var r = 0; r < configuration.EpochCount; r++)
                    {
                        for (var i = 0; i < inputs.Length; i++)
                        {
                            await Calculate(inputs[i]);
                            error = await BackPropagation(outputs[i], configuration.Speed);
                        }
                    }
                } while (error > configuration.TargetError && !cancellationToken.IsCancellationRequested);

                return error;
            }, cancellationToken);
            return task;
        }
    }
}