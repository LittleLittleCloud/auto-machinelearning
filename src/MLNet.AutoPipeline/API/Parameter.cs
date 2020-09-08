// <copyright file="Parameter.cs" company="BigMiao">
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
    /// Create a parameter that can be used in <see cref="SweepableOption{TOption}"/>.
    /// </summary>
    /// <typeparam name="T">type of Parameter.</typeparam>
    public class Parameter<T> : IParameter
    {
        public IValueGenerator ValueGenerator { get; private set; }

        internal Parameter(int min, int max, bool logBase = false, int steps = 100)
        {
            Contract.Assert(max > min);
            Contract.Assert(steps > 0);
            Contract.Assert(!logBase || (logBase && min > 0));
            var option = new Int32ValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new Int32ValueGenerator(option);
        }

        internal Parameter(long min, long max, bool logBase = false, int steps = 100)
        {
            Contract.Assert(max > min);
            Contract.Assert(steps > 0);
            Contract.Assert(!logBase || (logBase && min > 0));
            var option = new LongValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new LongValueGenerator(option);
        }

        internal Parameter(float min, float max, bool logBase = false, int steps = 100)
        {
            Contract.Assert(max > min);
            Contract.Assert(steps > 0);
            Contract.Assert(!logBase || (logBase && min > 0));
            var option = new FloatValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new FloatValueGenerator(option);
        }

        internal Parameter(double min, double max, bool logBase = false, int steps = 100)
        {
            Contract.Assert(max > min);
            Contract.Assert(steps > 0);
            Contract.Assert(!logBase || (logBase && min > 0));
            var option = new DoubleValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new DoubleValueGenerator(option);
        }

        internal Parameter(T[] candidates)
        {
            var option = new DiscreteValueGenerator<T>.Option<T>()
            {
                Values = candidates,
            };

            this.ValueGenerator = new DiscreteValueGenerator<T>(option);
        }
    }

    /// <summary>
    /// </summary>
    internal interface IParameter
    {
        IValueGenerator ValueGenerator { get; }
    }
}
