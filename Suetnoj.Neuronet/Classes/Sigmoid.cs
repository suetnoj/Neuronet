using Suetnoj.Neuronet.Interfaces;

namespace Suetnoj.Neuronet.Classes;

public class Sigmoid : IActivator
{
    public double Function(double value) => (1 / (1 + Math.Exp(-value)));

    public double Derivative(double value) => value * (1 - value);
}