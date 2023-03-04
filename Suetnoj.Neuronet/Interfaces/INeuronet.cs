using Suetnoj.Neuronet.Structs;

namespace Suetnoj.Neuronet.Interfaces;

public interface INeuronet
{
    public double[][] Neurons { get; }
    public double[][][] Synapces { get; }

    public IActivator Activator { get; }
    public Task<double[]> Calculate(double[] input);

    public Task<double> Teach(double[][] inputs, double[][] outputs, TrainingConfiguration configuration,
        CancellationToken cancellationToken);
}