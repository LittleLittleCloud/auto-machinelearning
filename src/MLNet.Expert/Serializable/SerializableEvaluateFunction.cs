// <copyright file="SerializableEvaluateFunction.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.AutoPipeline;
using System;

namespace MLNet.Expert.Serializable
{
    internal class SerializableEvaluateFunction
    {
        public delegate double EvaluateFunctionWithLabel(MLContext context, IDataView dataView, string label);

        public SerializableEvaluateFunction(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; }

        public EvaluateFunctionWithLabel Accuracy = (MLContext context, IDataView eval, string label) =>
        {
            return context.BinaryClassification.Evaluate(eval, label).Accuracy;
        };


    }
}
