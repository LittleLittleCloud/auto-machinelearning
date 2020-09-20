// <copyright file="SinglePipelineTrainingService.cs" company="BigMiao">
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

namespace MLNet.AutoPipeline
{
    internal class SinglePipelineTrainingService : ITrainingService
    {
        private MLContext context;
        private Option option;

        public SinglePipelineTrainingService(MLContext context, SingleEstimatorSweepablePipeline pipeline, Option option)
        {
            this.context = context;
            this.option = option;
            this.Pipeline = pipeline;
        }

        public SingleEstimatorSweepablePipeline Pipeline { get; }

        public ISweeper Sweeper
        {
            get => this.option.ParameterSweeper;
        }

        public EvaluateFunction EvaluateFunction
        {
            get => this.option.EvaluationMetric;
        }

        public IEnumerable<EvaluateFunction> MetricFunctions
        {
            get => this.option.Metrics;
        }

        public event EventHandler<IterationInfo> IterationInfoHandler;

        public Task<IterationInfo> StartTrainingAsync(IDataView train, IDataView validate, CancellationToken ct = default, IProgress<IterationInfo> reporter = null)
        {
            // TODO check schema
            return Task.Run(() =>
            {
                var stopWatch = new Stopwatch();
                IterationInfo bestIteration = null;
                foreach (var parameters in this.Sweeper.ProposeSweeps(this.Pipeline, this.option.ParameterSweepingIteration))
                {
                    ct.ThrowIfCancellationRequested();
                    stopWatch.Start();
                    var pipeline = this.Pipeline.BuildFromParameters(parameters);
                    var model = pipeline.Fit(train);
                    var eval = model.Transform(validate);
                    var evalScore = this.EvaluateFunction(this.context, eval);
                    var scores = this.MetricFunctions.Select(f => f(this.context, eval)).ToArray();
                    stopWatch.Stop();
                    var iteration = new IterationInfo(this.Pipeline, parameters, stopWatch.Elapsed.TotalSeconds, evalScore, this.option.IsMaximizng, scores, model);
                    reporter?.Report(iteration);
                    this.IterationInfoHandler?.Invoke(this, iteration);
                    if (bestIteration == null || iteration > bestIteration)
                    {
                        bestIteration = iteration;
                    }

                    // update sweeper
                    var runHistory = new RunResult(parameters, evalScore, this.option.IsMaximizng);
                    this.Sweeper.AddRunHistory(runHistory);
                }

                return bestIteration;
            });
        }

        public Task<IterationInfo> StartTrainingCVAsync(IDataView train, int fold = 5, CancellationToken ct = default, IProgress<IterationInfo> reporter = null)
        {
            throw new NotImplementedException();
        }

        public class Option
        {
            public ISweeper ParameterSweeper;

            public EvaluateFunction EvaluationMetric;

            public bool IsMaximizng;

            public EvaluateFunction[] Metrics;

            public int ParameterSweepingIteration = 100;
        }
    }
}
