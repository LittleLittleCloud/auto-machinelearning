// <copyright file="Experiment.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Runtime;
using MLNet.AutoPipeline.Metric;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection.Emit;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLNet.AutoPipeline
{
    public class Experiment
    {
        private Option option;
        private MLContext context;
        private IEnumerable<SweepablePipeline> sweepablePipelines;
        private double timeLeft;

        public Experiment(MLContext context, SweepablePipeline pipeline, Option option)
        {
            this.option = option;
            this.context = context;
            this.sweepablePipelines = new SweepablePipeline[1] { pipeline };
            this.timeLeft = option.MaximumTrainingTime;
        }

        public Experiment(MLContext context, EstimatorNodeChain estimatorNodeChain, Option option)
        {
            this.option = option;
            this.context = context;
            this.sweepablePipelines = estimatorNodeChain.BuildSweepablePipelines();
            this.timeLeft = option.MaximumTrainingTime;
        }

        /// <summary>
        /// Train experiment using <paramref name="train"/> and validate it on <paramref name="validate"/>.
        /// </summary>
        /// <param name="train">train dataset.</param>
        /// <param name="validate">validate dataset. this dataset is only used for hypeparameter optimization.</param>
        /// <param name="reporter">train-progress reporter. default is null.</param>
        /// <param name="ct">cancelation tokem, default is <see cref="CancellationToken.None"/>.</param>
        /// <returns>experiment result.</returns>
        public async Task<ExperimentResult> TrainAsync(IDataView train, IDataView validate, IProgress<IterationInfo> reporter = null, CancellationToken ct = default)
        {
            var experimentResult = new ExperimentResult();

            await Task.Run(() =>
            {
                foreach (var sweepablePipeline in this.sweepablePipelines)
                {
                    sweepablePipeline.UseSweeper(this.option.Sweeper.Clone() as ISweeper);
                    foreach (var sweepingInfo in sweepablePipeline.Sweeping(this.option.Iteration))
                    {
                        var stopWatch = new Stopwatch();
                        stopWatch.Start();

                        var pipeline = sweepingInfo.Pipeline;
                        var parameters = sweepingInfo.Parameters;
                        if (ct.IsCancellationRequested)
                        {
                            return;
                        }

                        // train
                        var model = pipeline.Fit(train);
                        var val_eval = model.Transform(validate);

                        // evaluate
                        var evaluateMetrics = new List<IterationInfo.Metric>();
                        foreach (var metric in this.option.Metrics)
                        {
                            var score = metric.Score(this.context, val_eval, this.option.Label);
                            evaluateMetrics.Add(new IterationInfo.Metric(metric.Name, score));
                        }

                        // score
                        var validateScoreMetric = new IterationInfo.Metric(this.option.ScoreMetric.Name, this.option.ScoreMetric.Score(this.context, val_eval, this.option.Label));

                        var iterationInfo = new IterationInfo(sweepablePipeline, sweepingInfo.Parameters, stopWatch.Elapsed.TotalSeconds, validateScoreMetric, evaluateMetrics, this.option.ScoreMetric.IsMaximizing);

                        // update sweeper
                        var runHistory = new RunResult(sweepingInfo.Parameters, validateScoreMetric.Score, iterationInfo.IsMetricMaximizing);
                        sweepablePipeline.Sweeper.AddRunHistory(runHistory);

                        // report
                        experimentResult.AddRunHistory(iterationInfo, model);
                        reporter.Report(iterationInfo);

                        stopWatch.Stop();
                        this.timeLeft -= stopWatch.Elapsed.TotalSeconds;
                        if (this.timeLeft < 0)
                        {
                            return;
                        }
                    }
                }
            });

            return experimentResult;
        }

        /// <summary>
        /// Train experiment using train-validate split, during training, <paramref name="train"/> will be split into train and validate dataset using <paramref name="validateFraction"/>.
        /// </summary>
        /// <param name="train">train dataset.</param>
        /// <param name="validateFraction">fraction of data used from <paramref name="train"/> to validate training, default is 0.3f.</param>
        /// <param name="reporter">train-progress reporter. default is null.</param>
        /// <param name="ct">cancelation tokem, default is <see cref="CancellationToken.None"/>.</param>
        /// <returns>experiment result.</returns>
        public async Task<ExperimentResult> TrainAsync(IDataView train, float validateFraction = 0.3f, IProgress<IterationInfo> reporter = null, CancellationToken ct = default)
        {
            if (validateFraction <= 0 || validateFraction >= 1)
            {
                throw new Exception("valudateFraction must be between 0 and 1");
            }

            var split = this.context.Data.TrainTestSplit(train, validateFraction);

            return await this.TrainAsync(split.TrainSet, split.TestSet, reporter, ct);
        }

        public class Option
        {
            /// <summary>
            /// Sweeper used for hypeparameter optimization. Default is <see cref="RandomGridSweeper"/>.
            /// </summary>
            public ISweeper Sweeper { get; set; } = new RandomGridSweeper(new RandomGridSweeper.Option());

            /// <summary>
            /// Number of iteration <see cref="Sweeper"/> will try in each sweeping process. Default is 100. This value will be used to set maximum parameter for <seealso cref="SweepablePipeline.Sweeping(int)"/>.
            /// </summary>
            public int Iteration { get; set; } = 100;

            /// <summary>
            /// Metric to optimize. This metric will be recorded in <see cref="IterationInfo.ScoreMetric"/> in each sweeping.
            /// </summary>
            public IMetric ScoreMetric { get; set; }

            /// <summary>
            /// Metrics to be evaluate during training. These metrics will be recorded in <see cref="IterationInfo."/> 
            /// </summary>
            public IEnumerable<IMetric> Metrics { get; set; } = new List<IMetric>();

            /// <summary>
            /// Label column name. Default is Label.
            /// </summary>
            public string Label { get; set; } = "Label";

            public double MaximumTrainingTime = 100;
        }
    }
}
