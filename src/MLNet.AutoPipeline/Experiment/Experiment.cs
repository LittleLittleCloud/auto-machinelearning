// <copyright file="Experiment.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Runtime;
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
    public delegate double EvaluateFunction(MLContext context, IDataView dataView);

    public class Experiment
    {
        private Option option;
        private MLContext context;
        private SweepablePipeline sweepablePipeline;
        private double timeLeft;
        private Logger logger = Logger.Instance;

        public Experiment(MLContext context, SweepablePipeline pipeline, Option option)
        {
            this.option = option;
            this.context = context;
            this.sweepablePipeline = pipeline;
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
            var trainingServiceOption = new SinglePipelineTrainingService.Option()
            {
                ParameterSweeper = this.option.ParameterSweeper,
                EvaluationMetric = this.option.EvaluateFunction,
                Metrics = this.option.Metrics,
                IsMaximizng = this.option.IsMaximizing,
                ParameterSweepingIteration = this.option.ParameterSweeperIteration,
            };

            foreach (var pipelineConfig in this.option.PipelineSweeper.ProposeSweeps(this.sweepablePipeline, this.option.PipelineSweeperIteration))
            {
                var singleSweepablePipeline = this.sweepablePipeline.BuildFromParameters(pipelineConfig);
                var trainingService = new SinglePipelineTrainingService(this.context, singleSweepablePipeline, trainingServiceOption);
                var bestIteration = await trainingService.StartTrainingAsync(train, validate, ct, reporter);
                this.timeLeft -= bestIteration.TrainingTime;
                var bestModel = bestIteration.Model;
                if (this.timeLeft < 0)
                {
                    this.logger.Trace(MessageSensitivity.All, "experiment time out");
                    break;
                }

                experimentResult.AddRunHistory(bestIteration, bestModel);

                // TODO
                // Add Run history to this.option.PipelineSweeper.
            }

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
            public ISweeper ParameterSweeper { get; set; } = new RandomGridSweeper();

            /// <summary>
            /// Sweeper used for sweeping over pipeline when there are more than one transformers or trainers in a single pipe. Default is <see cref="RandomGridSweeper"/>.
            /// </summary>
            public ISweeper PipelineSweeper { get; set; } = new GridSearchSweeper();

            /// <summary>
            /// Number of iteration <see cref="ParameterSweeper"/> will try in each hypeparameter optimization process. This value will be used to set maximum parameter for <seealso cref="SweepablePipeline.Sweeping(int)"/>. Default is 100. 
            /// </summary>
            public int ParameterSweeperIteration { get; set; } = 100;

            /// <summary>
            /// The maximum number of pipelines the ongoing experiment will try. This value will be used to set maximum parameter for <seealso cref="SweepablePipeline.Sweeping(int)"/>. Default is 100.
            /// </summary>
            public int PipelineSweeperIteration { get; set; } = 100;

            /// <summary>
            /// Inidcate whether the valudating score from <see cref="EvaluateFunction"/> should be maximize during training. Default to true.
            /// </summary>
            public bool IsMaximizing { get; set; } = true;

            /// <summary>
            /// Indicate how to evaluate validate score during training. The function must match <see cref="AutoPipeline.EvaluateFunction"/> signature.
            /// </summary>
            public EvaluateFunction EvaluateFunction { get; set; }

            public EvaluateFunction[] Metrics { get; set; }

            public double MaximumTrainingTime = 100;
        }
    }
}
