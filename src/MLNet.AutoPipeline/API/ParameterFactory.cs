// <copyright file="ParameterFactory.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Provides methods to create <see cref="Parameter{T}"/>.
    /// </summary>
    public static class ParameterFactory
    {
        /// <summary>
        /// Create a sweepable parameter with type Int32.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="Parameter{T}"/> where T is int.</returns>
        public static Parameter<int> CreateInt32Parameter(int min, int max, bool logBase = false, int steps = 100)
        {
            return new Parameter<int>(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Long.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="Parameter{T}"/> where T is long.</returns>
        public static Parameter<long> CreateLongParameter(long min, long max, bool logBase = false, int steps = 100)
        {
            return new Parameter<long>(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Float.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="Parameter{T}"/> where T is float.</returns>
        public static Parameter<float> CreateFloatParameter(float min, float max, bool logBase = false, int steps = 100)
        {
            return new Parameter<float>(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Double.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="Parameter{T}"/> where T is double.</returns>
        public static Parameter<double> CreateDoubleParameter(double min, double max, bool logBase = false, int steps = 100)
        {
            return new Parameter<double>(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with discrete values.
        /// </summary>
        /// <typeparam name="T">type of values.</typeparam>
        /// <param name="objects">discrete values.</param>
        /// <returns><see cref="Parameter{T}"/>.</returns>
        public static Parameter<T> CreateDiscreteParameter<T>(params T[] objects)
        {
            return new Parameter<T>(objects);
        }

        public static Parameter<T> CreateDiscreteParameter<T>(T objects)
        {
            return new Parameter<T>(new T[] { objects });
        }
    }
}
