// <copyright file="ISweepableEstimator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using Newtonsoft.Json;

namespace MLNet.AutoPipeline
{
    public interface ISweepableEstimator
    {
        TransformerScope Scope { get; }

        string EstimatorName { get; }

        string[] InputColumns { get; }

        string[] OutputColumns { get; }
    }

    public interface ISweepableEstimator<out TTrain, TOption> : ISweepableEstimator
        where TTrain : IEstimator<ITransformer>
        where TOption : class
    {
        public Func<TOption, TTrain> EstimatorFactory { get; }

        public SweepableOption<TOption> OptionBuilder { get; }
    }

    [DataContract]
    internal class CodeGenNodeContract
    {
        [DataMember(Name = "parameters")]
        public Parameters Parameters { get; set; }

        [DataMember(Name = "name", IsRequired = true)]
        public string EstimatorName { get; set; }

        [DataMember(Name = "input_column", IsRequired = true)]
        public string[] InputColumns { get; set; }

        [DataMember(Name = "output_column", IsRequired = true)]
        public string[] OutputColumns { get; set; }

        public override string ToString()
        {
            return JsonConvert.SerializeObject(this, Formatting.Indented);
        }
    }
}
