// <copyright file="SweepableOption.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public abstract class SweepableOption<TOption>
        where TOption : class
    {
        private readonly HashSet<string> _ids = new HashSet<string>();

        private TOption defaultOption = null;

        public IValueGenerator[] ValueGenerators { get => this.GetValueGenerators(); }

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

            // set up sweepable parameters
            foreach (var generator in this.ValueGenerators)
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

        /// <summary>
        /// Build option using <paramref name="parameters"/>.
        /// </summary>
        /// <param name="parameters">a set of parameters used to build option.</param>
        /// <returns><paramref name="TOption"/>.</returns>
        public TOption BuildOption(ParameterSet parameters)
        {
            var option = this.CreateDefaultOption();
            foreach (var param in parameters)
            {
                if (this._ids.Contains(param.ID))
                {
                    var value = param.RawValue;
                    typeof(TOption).GetField(param.Name)?.SetValue(option, value);
                }
            }

            return option;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Type of option: {typeof(TOption).Name}");
            sb.AppendLine();
            foreach (var value in this.ValueGenerators)
            {
                sb.AppendLine(value.ToString());
            }

            return sb.ToString();
        }

        private Dictionary<string, Parameter> GetSweepableParameterValue()
        {
            var paramaters = this.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                     .Where(x => Attribute.GetCustomAttribute(x, typeof(ParameterAttribute)) != null)
                     .Where(x => x.FieldType.IsSubclassOf(typeof(Parameter)) || x.FieldType == typeof(Parameter));

            var paramatersDictionary = new Dictionary<string, Parameter>();

            foreach (var param in paramaters)
            {
                var paramaterAttribute = Attribute.GetCustomAttribute(param, typeof(ParameterAttribute)) as ParameterAttribute;
                var paramValue = (Parameter)param.GetValue(this);
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
    }
}
