// <copyright file="TrainingManager.cs" company="BigMiao">
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
    internal class TrainingManager
    {
        private IEnumerable<Column> columns;
        private MLContext context;
        private Option option;
        private PipelineBuilder pipelineBuilder;
        private Dictionary<IDictionary<string, string>, ITrainingService> singlePipelineTrainingServiceMap;
        private Dictionary<IDictionary<string, string>, IterationInfo> bestIterations;
        private ISweeper pipelineSweeper;

        public TrainingManager(MLContext context, IEnumerable<Column> columns, Option option)
        {
            this.context = context;
            this.columns = columns;
            this.singlePipelineTrainingServiceMap = new Dictionary<IDictionary<string, string>, ITrainingService>();
            this.bestIterations = new Dictionary<IDictionary<string, string>, IterationInfo>();
            this.option = option;
            this.pipelineSweeper = context.AutoML().Serializable().Factory.CreateSweeper(this.option.ParameterSweeper);
            this.pipelineBuilder = new PipelineBuilder(this.option.TaskType, this.option.IsAzureAttach, true);
            this.Pipeline = this.pipelineBuilder.BuildPipeline(context, columns);
        }

        public async Task<IterationInfo> StartTrainingAsync(IDataView train, IDataView validate, CancellationToken ct = default, IProgress<IterationInfo> reporter = null)
        {
            var timeLeft = this.option.MaximumTrainingTime;
            var stopWatch = new Stopwatch();
            foreach (var parameter in this.pipelineSweeper.ProposeSweeps(this.Pipeline, this.option.PipelineSweepingIteration))
            {
                ct.ThrowIfCancellationRequested();
                if (!this.singlePipelineTrainingServiceMap.ContainsKey(parameter))
                {
                    // if single pipeline is new, create training service and add it to map.
                    var singleSweepablePipeline = this.Pipeline.BuildFromParameters(parameter);
                    this.singlePipelineTrainingServiceMap[parameter] = this.CreateTrainingService(singleSweepablePipeline);
                }

                stopWatch.Start();
                var trainingService = this.singlePipelineTrainingServiceMap[parameter];
                var bestIteration = await trainingService.StartTrainingAsync(train, validate, ct, reporter);
                stopWatch.Stop();

                // update best iteration
                if (!this.bestIterations.ContainsKey(parameter) || this.bestIterations[parameter] < bestIteration)
                {
                    this.bestIterations[parameter] = bestIteration;
                }

                timeLeft -= stopWatch.Elapsed.TotalSeconds;
                if (timeLeft < 0)
                {
                    break;
                }
            }

            if (this.bestIterations.Count() == 0)
            {
                throw new Exception("No complete iteration found, please try to extend the training time");
            }

            return this.bestIterations.Values.Max();
        }

        public ITrainingService CreateTrainingService(SingleEstimatorSweepablePipeline pipeline)
        {
            if (!this.option.IsAzureAttach && !this.option.IsNNITraining)
            {
                var option = new LocalTrainingService.Option()
                {
                    ParameterSweeper = this.option.ParameterSweeper,
                    EvaluationMetric = this.option.EvaluationMetric,
                    IsMaximizng = this.option.IsMaximizng,
                    Metrics = this.option.Metrics,
                    ParameterSweepingIteration = this.option.ParameterSweepingIteration,
                    Label = this.option.Label,
                };
                return new LocalTrainingService(this.context, pipeline, option);
            }

            throw new Exception("Not implemented");
        }

        public SweepablePipeline Pipeline { get; }

        public class Option
        {
            public string PipelineSweeper;

            public int PipelineSweepingIteration = 20;

            public string ParameterSweeper;

            public string EvaluationMetric;

            public bool IsMaximizng;

            public string[] Metrics;

            public int ParameterSweepingIteration = 100;

            public double MaximumTrainingTime = 100;

            public bool IsAzureAttach = false;

            public bool IsNNITraining = false;

            public string Label;

            public TaskType TaskType;
        }
    }
}
