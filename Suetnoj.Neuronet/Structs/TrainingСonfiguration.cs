namespace Suetnoj.Neuronet.Structs;

public struct TrainingConfiguration
{
    public TrainingConfiguration(double speed, int epochCount, double targetError)
    {
        Speed = speed;
        EpochCount = epochCount;
        TargetError = targetError;
    }

    public double Speed { get;}
    public int EpochCount { get;}
    public double TargetError { get;}
}