// <copyright file="INode.cs" company="BigMiao">
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
    public enum NodeType
    {
        /// <summary>
        /// Sweepable node Type.
        /// </summary>
        Sweepable = 0,

        /// <summary>
        /// Unsweepable node type.
        /// </summary>
        Unsweeapble = 1,
    }

    public interface INode
    {
        IEstimator<ITransformer> BuildEstimator(ParameterSet parameters = null);

        TransformerScope Scope { get; }

        IValueGenerator[] ValueGenerators { get; }

        NodeType NodeType { get; }

        string Summary();

        string EstimatorName { get; }

        string[] InputColumns { get; }

        string[] OutputColumns { get; }
    }

    public interface INode<TTrain> : INode
        where TTrain : IEstimator<ITransformer>
    {
        TTrain BuildEstimator(ParameterSet parameters = null);
    }

    [DataContract]
    internal class CodeGenNodeContract
    {
        [DataMember(Name = "parameters")]
        public ParameterSet Parameters { get; set; }

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
