// <copyright file="ParameterAttribute.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    internal class ParameterAttribute : Attribute
    {
        private IList _value;
        private Type _meta;

        public ParameterAttribute(string name, int min, int max, int step = 1)
        {
            this._meta = typeof(int);
            var intList = new List<int>();
            for (var i = min; i <= max; i += step)
            {
                intList.Add(i);
            }

            intList.Add(max);
            this._value = intList;

            var option = new IntParamOptions()
            {
                Name = name,
                Min = min,
                Max = max,
                StepSize = step,
            };

            this.ValueGenerator = new Int32ValueGenerator(option);
        }

        public ParameterAttribute(string name, float min, float max, float step = 1f)
        {
            this._meta = typeof(float);
            var intList = new List<float>();
            for (var i = min; i <= max; i += step)
            {
                intList.Add(i);
            }

            intList.Add(max);
            this._value = intList;

            var option = new FloatParamOptions()
            {
                Name = name,
                Min = min,
                Max = max,
                StepSize = step,
            };

            this.ValueGenerator = new FloatValueGenerator(option);
        }

        public ParameterAttribute(string name, string[] candidates)
        {
            this._meta = typeof(string);
            this._value = candidates.ToList();

            var option = new DiscreteParamOptions()
            {
                Name = name,
                Values = candidates,
            };

            this.ValueGenerator = new DiscreteValueGenerator(option);
        }

        public IList Value => this._value;

        public Type Meta => this._meta;

        public IValueGenerator ValueGenerator { get; }

    }
}
