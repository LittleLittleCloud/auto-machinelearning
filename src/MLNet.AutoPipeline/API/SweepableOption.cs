// <copyright file="SweepableOption.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public abstract class SweepableOption<TOption> : ISweepable<TOption>
        where TOption : class
    {
        private readonly HashSet<string> _ids = new HashSet<string>();

        private TOption defaultOption = null;

        public IEnumerable<IValueGenerator> SweepableValueGenerators { get => this.GetValueGenerators(); }

        public SweepableOption()
        {
        }

        public SweepableOption(TOption option)
            : this()
        {
            this.defaultOption = option;
        }

        public TOption CreateDefaultOption()
        {
            var assem = typeof(TOption).Assembly;
            var option = assem.CreateInstance(typeof(TOption).FullName) as TOption;

            var paramaters = this.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                     .Where(x => Attribute.GetCustomAttribute(x, typeof(ParameterAttribute)) != null)
                     .Where(x => !(x.GetValue(this) is IParameter));

            foreach (var parameter in paramaters)
            {
                var parameterAttribute = Attribute.GetCustomAttribute(parameter, typeof(ParameterAttribute)) as ParameterAttribute;
                var parameterName = parameterAttribute.Name ?? parameter.Name;
                var destFieldType = option.GetType().GetField(parameterName)?.FieldType;
                if (parameter.FieldType == destFieldType)
                {
                    option.GetType().GetField(parameterName).SetValue(option, parameter.GetValue(this));
                }
            }

            // set up sweepable parameters
            foreach (var generator in this.SweepableValueGenerators)
            {
                var param = generator.CreateFromNormalized(0);
                var value = param.RawValue;
                option.GetType().GetField(param.Name)?.SetValue(option, value);
            }

            // use field in defaultOption
            if (this.defaultOption != null)
            {
                Util.CopyFieldsTo(this.defaultOption, option);
            }

            return option;
        }

        public void SetDefaultOption(TOption option)
        {
            if (option != null)
            {
                this.defaultOption = option;
            }
        }

        public TOption BuildFromParameters(IDictionary<string, string> parameters)
        {
            var option = this.CreateDefaultOption();

            foreach (var generator in this.SweepableValueGenerators)
            {
                if (parameters.ContainsKey(generator.ID))
                {
                    var valueText = parameters[generator.ID];
                    var rawValue = generator.CreateFromString(valueText).RawValue;

                    typeof(TOption).GetField(generator.Name)?.SetValue(option, rawValue);
                }
            }

            return option;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Type of option: {typeof(TOption).Name}");
            sb.AppendLine();
            foreach (var value in this.SweepableValueGenerators)
            {
                sb.AppendLine(value.ToString());
            }

            return sb.ToString();
        }

        private Dictionary<string, IParameter> GetSweepableParameterValue()
        {
            var paramaters = this.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                     .Where(x => Attribute.GetCustomAttribute(x, typeof(ParameterAttribute)) != null)
                     .Where(x => x.GetValue(this) is IParameter);

            var paramatersDictionary = new Dictionary<string, IParameter>();

            foreach (var param in paramaters)
            {
                var paramaterAttribute = Attribute.GetCustomAttribute(param, typeof(ParameterAttribute)) as ParameterAttribute;
                var paramValue = (IParameter)param.GetValue(this);
                if (paramValue == null)
                {
                    // TODO: add warning for wrong parameter type.
                    continue;
                }

                if (paramaterAttribute.Name != null)
                {
                    paramValue.ValueGenerator.Name = paramaterAttribute.Name;
                }
                else
                {
                    paramValue.ValueGenerator.Name = param.Name;
                }

                paramatersDictionary.Add(param.Name, paramValue);
            }

            return paramatersDictionary;
        }

        private IValueGenerator[] GetValueGenerators()
        {
            var valueGenerators = this.GetSweepableParameterValue().Select(kv =>
            {
                return kv.Value.ValueGenerator;
            }).ToArray();

            foreach (var valueGenerator in valueGenerators)
            {
                this._ids.Add(valueGenerator.ID);
            }

            return valueGenerators;
        }

        /// <summary>
        /// Create a sweepable parameter with type Int32.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="IParameter"/>.</returns>
        public static Parameter<int> CreateInt32Parameter(int min, int max, bool logBase = false, int steps = 100)
        {
            return ParameterFactory.CreateInt32Parameter(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Long.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="IParameter"/>.</returns>
        public static Parameter<long> CreateLongParameter(long min, long max, bool logBase = false, int steps = 100)
        {
            return ParameterFactory.CreateLongParameter(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Float.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="IParameter"/>.</returns>
        public static Parameter<float> CreateFloatParameter(float min, float max, bool logBase = false, int steps = 100)
        {
            return ParameterFactory.CreateFloatParameter(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with type Double.
        /// </summary>
        /// <param name="min">min value.</param>
        /// <param name="max">max value.</param>
        /// <param name="logBase">log base.</param>
        /// <param name="steps">steps.</param>
        /// <returns><see cref="IParameter"/>.</returns>
        public static Parameter<double> CreateDoubleParameter(double min, double max, bool logBase = false, int steps = 100)
        {
            return ParameterFactory.CreateDoubleParameter(min, max, logBase, steps);
        }

        /// <summary>
        /// Create a sweepable parameter with discrete values.
        /// </summary>
        /// <typeparam name="T">type of values.</typeparam>
        /// <param name="objects">discrete values.</param>
        /// <returns><see cref="Parameter{T}"/>.</returns>
        public static Parameter<T> CreateDiscreteParameter<T>(params T[] objects)
        {
            return ParameterFactory.CreateDiscreteParameter<T>(objects);
        }

        /// <summary>
        /// Create a sweepable parameter with discrete values.
        /// </summary>
        /// <typeparam name="T">type of values.</typeparam>
        /// <param name="objects">discrete values.</param>
        /// <returns><see cref="Parameter{T}"/>.</returns>
        public static Parameter<T> CreateDiscreteParameter<T>(T objects)
        {
            return ParameterFactory.CreateDiscreteParameter(objects);
        }
    }
}
