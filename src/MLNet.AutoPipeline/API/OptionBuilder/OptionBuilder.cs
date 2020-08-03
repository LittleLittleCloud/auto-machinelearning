// <copyright file="OptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public abstract class OptionBuilder<TOption>
        where TOption : class
    {
        private readonly HashSet<string> _ids = new HashSet<string>();

        private TOption defaultOption = null;

        public IValueGenerator[] ValueGenerators { get; private set; }

        public IParameterValue[] UnsweepableParameters { get => this.GetUnsweepableParameterValues(); }

        public OptionBuilder()
        {
            this.ValueGenerators = this.GetValueGenerators();
        }

        public OptionBuilder(TOption option)
            : this()
        {
            this.defaultOption = option;
        }

        public TOption CreateDefaultOption()
        {
            var assem = typeof(TOption).Assembly;
            var option = assem.CreateInstance(typeof(TOption).FullName) as TOption;

            // set up unsweepable parameters
            foreach (var param in this.UnsweepableParameters)
            {
                var value = param.RawValue;
                option.GetType().GetField(param.Name)?.SetValue(option, value);
            }

            // set up sweeppable parameters
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

        private Dictionary<string, SweepableParameterAttribute> GetSweepableParameterAttributes()
        {
            var paramaters = this.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                     .Where(x => Attribute.GetCustomAttribute(x, typeof(SweepableParameterAttribute)) != null);

            var paramatersDictionary = new Dictionary<string, SweepableParameterAttribute>();

            foreach (var param in paramaters)
            {
                var paramaterAttribute = Attribute.GetCustomAttribute(param, typeof(SweepableParameterAttribute)) as SweepableParameterAttribute;
                paramatersDictionary.Add(param.Name, paramaterAttribute);
            }

            return paramatersDictionary;
        }

        private IValueGenerator[] GetValueGenerators()
        {
            var valueGenerators = this.GetSweepableParameterAttributes().Select(kv =>
            {
                kv.Value.ValueGenerator.Name = kv.Key;
                return kv.Value.ValueGenerator;
            }).ToArray();

            foreach (var valueGenerator in valueGenerators)
            {
                this._ids.Add(valueGenerator.ID);
            }

            return valueGenerators;
        }

        private IParameterValue[] GetUnsweepableParameterValues()
        {
            var parameters = this.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance)
                                 .Where(x => Attribute.GetCustomAttribute(x, typeof(ParameterAttribute)) != null);

            var res = new List<ObjectParameterValue>();

            foreach (var parm in parameters)
            {
                var guid = Guid.NewGuid().ToString();
                var value = parm.GetValue(this);
                var paramValue = new ObjectParameterValue(parm.Name, parm.GetValue(this), guid);
                res.Add(paramValue);
            }

            return res.ToArray();
        }
    }
}
