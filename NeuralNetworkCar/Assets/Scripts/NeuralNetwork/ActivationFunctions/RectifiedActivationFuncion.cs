﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.ActivationFunctions
{
    public class RectifiedActivationFuncion : IActivationFunction
    {
        public double CalculateDerivative(double input)
        {
            return Math.Max(0, input);
        }

        public double CalculateOutput(double input)
        {
            return Math.Max(0, input);
        }
    }
}
