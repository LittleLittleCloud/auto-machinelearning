// <copyright file="LbfgsMaximumEntropyOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Sweepable option for <see cref="LbfgsMaximumEntropyMulticlassTrainer"/>.
    /// </summary>
    public class LbfgsMaximumEntropyOptionBuilder : OptionBuilder<LbfgsMaximumEntropyMulticlassTrainer.Options>
    {
        /// <summary>
        /// The L2 regularization hyperparameter.
        /// <para>Default sweeping configuration.</para>
        /// <list type="bullet">
        /// <item>
        /// <term>type</term>
        /// <description><c>Float</c></description>
        /// </item>
        /// <item>
        /// <term>minimum value</term>
        /// <description><c>1E-4F</c></description>
        /// </item>
        /// <item>
        /// <term>maximum value</term>
        /// <description><c>10F</c></description>
        /// </item>
        /// <item>
        /// <term>log base</term>
        /// <description><c>true</c></description>
        /// </item>
        /// <item>
        /// <term>step</term>
        /// <description><c>20</c></description>
        /// </item>
        /// </list>
        /// </summary>
        [SweepableParameter]
        public SweepableParameter L2Regularization = SweepableParameter.CreateFloatParameter(1E-4F, 10f, true, 20);

        /// <summary>
        /// The L1 regularization hyperparameter.
        /// <para>Default sweeping configuration.</para>
        /// <list type="bullet">
        /// <item>
        /// <term>type</term>
        /// <description><c>Float</c></description>
        /// </item>
        /// <item>
        /// <term>minimum value</term>
        /// <description><c>1E-4F</c></description>
        /// </item>
        /// <item>
        /// <term>maximum value</term>
        /// <description><c>10F</c></description>
        /// </item>
        /// <item>
        /// <term>log base</term>
        /// <description><c>true</c></description>
        /// </item>
        /// <item>
        /// <term>step</term>
        /// <description><c>20</c></description>
        /// </item>
        /// </list>
        /// </summary>
        [SweepableParameter]
        public SweepableParameter L1Regularization = SweepableParameter.CreateFloatParameter(1E-4F, 10f, true, 20);

        /// <summary>
        /// Memory size for <see cref="SdcaMaximumEntropyMulticlassTrainer"/> Low=faster, less accurate.
        /// <para>Default sweeping configuration.</para>
        /// <list type="bullet">
        /// <item>
        /// <term>type</term>
        /// <description><c>Int</c></description>
        /// </item>
        /// <item>
        /// <term>minimum value</term>
        /// <description><c>1</c></description>
        /// </item>
        /// <item>
        /// <term>maximum value</term>
        /// <description><c>1000</c></description>
        /// </item>
        /// <item>
        /// <term>log base</term>
        /// <description><c>true</c></description>
        /// </item>
        /// <item>
        /// <term>step</term>
        /// <description><c>20</c></description>
        /// </item>
        /// </list>
        /// </summary>
        [SweepableParameter]
        public SweepableParameter HistorySize = SweepableParameter.CreateInt32Parameter(1, 1000, true, 20);

        /// <summary>
        /// The name of the example weight column, default is null.
        /// </summary>
        [Parameter]
        public string ExampleWeightColumnName = default;

        /// <summary>
        /// Threshold for optimizer convergence, default is 1E-7F.
        /// </summary>
        [Parameter]
        public float OptimizationTolerance = 1E-7F;

        /// <summary>
        /// Enforce non-negative weights, default is false.
        /// </summary>
        [Parameter]
        public bool EnforceNonNegativity = false;

        internal static LbfgsMaximumEntropyOptionBuilder Default = new LbfgsMaximumEntropyOptionBuilder();
    }
}
