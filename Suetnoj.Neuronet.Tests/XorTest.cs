using Suetnoj.Neuronet.Classes;
using Suetnoj.Neuronet.Interfaces;

namespace Suetnoj.Neuronet.Tests;

public class XorTest
{
    [Theory]
    [InlineData(0, 0, 0, 0)]
    [InlineData(0, 0, 1, 1)]
    [InlineData(0, 1, 0, 1)]
    [InlineData(0, 1, 1, 0)]
    [InlineData(1, 0, 0, 1)]
    [InlineData(1, 0, 1, 0)]
    [InlineData(1, 1, 0, 0)]
    [InlineData(1, 1, 1, 1)]
    public async void Test(int x1, int x2, int x3, int y)
    {
        var network = new Perceptron(new[] { 3, 4, 4, 8 }, new Sigmoid());

        var result = await network.Calculate(new[] { (double)x1, x2, x3 });
        var rounded = result[0] >= 0.5 ? 1 : 0;

        Assert.True(rounded == y);
    }
}