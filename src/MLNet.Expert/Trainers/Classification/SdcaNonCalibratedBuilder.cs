// <copyright file="SdcaNonCalibratedBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert.Trainers.Classification
{
    /// <summary>
    /// Create a <see cref="Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer"/> with sweepable option.
    /// </summary>
    public class SdcaNonCalibratedBuilder : ICanCreateTrainer
    {
        private Option _option;
        private MLContext _context;
        private static SdcaNonCalibratedBuilder _instance = new SdcaNonCalibratedBuilder();

        public SdcaNonCalibratedBuilder(MLContext context, Option option)
        {
            this._context = context;
            this._option = option;
        }

        internal SdcaNonCalibratedBuilder()
        {
            this._option = new Option();
        }

        internal static SdcaNonCalibratedBuilder Instance
        {
            get => SdcaNonCalibratedBuilder._instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            Func<Option, SdcaNonCalibratedMulticlassTrainer> sdca = (option) =>
            {
                ISupportSdcaClassificationLoss lossFunction;
                if (option.LossFunction == LossFunctionType.HingeLoss)
                {
                    lossFunction = new HingeLoss(1);
                }
                else if (option.LossFunction == LossFunctionType.SmoothedHingeLoss)
                {
                    lossFunction = new SmoothedHingeLoss(1);
                }
                else
                {
                    lossFunction = new LogLoss();
                }

                return context.MulticlassClassification.Trainers.SdcaNonCalibrated(label, feature, l1Regularization: option.L1Reegularization, l2Regularization: option.L2Regularization, lossFunction: lossFunction);
            };

            var node = Util.CreateSweepableNode(sdca, this._option, estimatorName: "SdcaNonCalibrated");
            return Util.CreateEstimatorSingleNode(node);
        }

        /// <summary>
        /// Sweepable option for <see cref="SdcaMaximumEntropyBuilder"/>.
        /// </summary>
        public class Option : SweepableOption<Option>
        {
            [SweepableParameter(1E-4F, 10f, true, 20)]
            public float L2Regularization;

            [SweepableParameter(1E-4F, 10f, true, 20)]
            public float L1Reegularization;

            [SweepableParameter(new object[] { LossFunctionType.LogLoss, LossFunctionType.HingeLoss, LossFunctionType.SmoothedHingeLoss })]
            internal LossFunctionType LossFunction;
        }

        internal enum LossFunctionType
        {
            /// <summary>
            /// <see cref="Microsoft.ML.Trainers.LogLoss"/>
            /// </summary>
            LogLoss = 0,

            /// <summary>
            /// <see cref="Microsoft.ML.Trainers.HingeLoss"/>
            /// </summary>
            HingeLoss = 1,

            /// <summary>
            /// <see cref="Microsoft.ML.Trainers.SmoothedHingeLoss"/>
            /// </summary>
            SmoothedHingeLoss = 2,
        }
    }
}
