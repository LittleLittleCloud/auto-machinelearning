// <copyright file="ParameterAttribute.cs" company="BigMiao">
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
    /// Create sweepable parameter that can be used in <see cref="OptionBuilder{TOption}"/>.
    /// </summary>
    public class SweepableParameter
    {
        /// <summary>
        /// Create a sweepable parameter with type Int32.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="SweepableParameter"/>.</returns>
        public static SweepableParameter CreateInt32Parameter(int min, int max, bool logBase = false, int steps = 100)
        {
            return new SweepableParameter(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Long.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="SweepableParameter"/>.</returns>
        public static SweepableParameter CreateLongParameter(long min, long max, bool logBase = false, int steps = 100)
        {
            return new SweepableParameter(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Float.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="SweepableParameter"/>.</returns>
        public static SweepableParameter CreateFloatParameter(float min, float max, bool logBase = false, int steps = 100)
        {
            return new SweepableParameter(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Double.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="SweepableParameter"/>.</returns>
        public static SweepableParameter CreateDoubleParameter(double min, double max, bool logBase = false, int steps = 100)
        {
            return new SweepableParameter(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with discrete values.
        /// </summary>
        /// <param name="objects">discrete values.</param>
        /// <returns><see cref="SweepableParameter"/>.</returns>
        public static SweepableParameter CreateDiscreteParameter(object[] objects)
        {
            return new SweepableParameter(objects);
        }

        internal SweepableParameter(int min, int max, bool logBase = false, int steps = 100)
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

        internal SweepableParameter(long min, long max, bool logBase = false, int steps = 100)
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

        internal SweepableParameter(float min, float max, bool logBase = false, int steps = 100)
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

        internal SweepableParameter(double min, double max, bool logBase = false, int steps = 100)
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

        internal SweepableParameter(object[] candidates)
        {
            var option = new DiscreteValueGenerator.Option()
            {
                Values = candidates,
            };

            this.ValueGenerator = new DiscreteValueGenerator(option);
        }

        internal IValueGenerator ValueGenerator { get; }
    }
}
