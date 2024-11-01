using System;
using System.Security.Cryptography;

namespace NeuralNetwork.Utils
{
    public static class CryptoRandom
    {
        private static readonly Random _random;

        static CryptoRandom()
        {
            using (RNGCryptoServiceProvider p = new RNGCryptoServiceProvider())
            {
                _random = new Random(p.GetHashCode());
            }
        }

        public static double NextDouble()
        {
            return _random.NextDouble();
        }

        public static double NextDouble(double minValue, double maxValue)
        {
            if (minValue > maxValue)
                throw new ArgumentOutOfRangeException(nameof(minValue),
                "Минимальное значение должно быть меньше максимального.");

            return _random.NextDouble() * (maxValue - minValue) + minValue;
        }
    }
}
