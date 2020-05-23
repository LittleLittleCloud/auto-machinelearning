// <copyright file="Estimator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;

namespace MLNet.CodeGenerator
{
    public enum EstimatorType
    {
        /// <summary>
        /// ML.Net trainer.
        /// </summary>
        Trainer = 1,

        /// <summary>
        /// ML.Net transformer.
        /// </summary>
        Transformer = 2,

        /// <summary>
        /// ML.Net custom transformer.
        /// </summary>
        Custom = 3,
    }

    public class Estimator : ICodeGenNode
    {
        public string EstimatorName { get; private set; }

        public string MLContextName { get; } = "context";

        public string Prefix { get; private set; }

        public EstimatorType Type { get; private set; }

        public ParamaterList Paramaters { get; private set; }

        public Estimator(string estimatorName, string prefix, EstimatorType type, ParamaterList paramaters)
        {
            this.EstimatorName = estimatorName;
            this.Prefix = prefix;
            this.Type = type;
            this.Paramaters = paramaters;
        }

        public string GeneratorCode()
        {
            if (this.Type == EstimatorType.Custom)
            {
                throw new Exception("Custom transformer is not supported yet");
            }

            return $"{this.MLContextName}.{this.Prefix}.{this.EstimatorName}({this.Paramaters.GeneratorCode()})";
        }
    }
}
