﻿// <copyright file="LocalTrainingService.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.AutoPipeline;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLNet.Expert
{
    internal class LocalTrainingService : ITrainingService
    {
        private MLContext context;
        private Option option;

        public LocalTrainingService(MLContext context, SingleEstimatorSweepablePipeline pipeline, Option option)
        {
            this.context = context;
            this.option = option;
            this.Pipeline = pipeline;
        }

        public SingleEstimatorSweepablePipeline Pipeline { get; }

        public async Task<IterationInfo> StartTrainingAsync(IDataView train, IDataView validate, CancellationToken ct = default, IProgress<IterationInfo> reporter = null)
        {
            // TODO check schema
            var option = new SinglePipelineTrainingService.Option()
            {
                IsMaximizng = this.option.IsMaximizing,
                ParameterSweepingIteration = this.option.ParameterSweeperIteration,
                ParameterSweeper = this.context.AutoML().Serializable().Factory.CreateSweeper(this.option.ParameterSweeper),
                EvaluationMetric = this.CreateEvaluateFunction(this.option.EvaluationMetric),
                Metrics = this.option.Metrics.Select(metric => this.CreateEvaluateFunction(metric)).ToArray(),
            };
            var trainingService = new SinglePipelineTrainingService(this.context, this.Pipeline, option);
            return await trainingService.StartTrainingAsync(train, validate, ct, reporter);
        }

        public Task<IterationInfo> StartTrainingCVAsync(IDataView train, int fold = 5, CancellationToken ct = default, IProgress<IterationInfo> reporter = null)
        {
            throw new NotImplementedException();
        }

        private EvaluateFunction CreateEvaluateFunction(string metric)
        {
            return (MLContext context, IDataView dataView) =>
            {
                return context.AutoML().Serializable().Factory.CreateEvaluateFunction(metric)(context, dataView, this.option.Label);
            };
        }

        public class Option
        {
            public string ParameterSweeper;

            public string EvaluationMetric;

            public bool IsMaximizing;

            public string[] Metrics;

            public int ParameterSweeperIteration = 100;

            public double MaximumTrainingTime = 100;

            public string Label;
        }
    }
}
