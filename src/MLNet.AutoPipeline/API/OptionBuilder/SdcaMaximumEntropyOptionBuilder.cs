// <copyright file="SdcaMaximumEntropyOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline.API.OptionBuilder
{
    /// <summary>
    /// Sweepable option for <see cref="SdcaMaximumEntropyMulticlassTrainer"/>.
    /// </summary>
    public sealed class SdcaMaximumEntropyOptionBuilder : OptionBuilder<SdcaMaximumEntropyMulticlassTrainer.Options>
    {
        internal static SdcaMaximumEntropyOptionBuilder Default = new SdcaMaximumEntropyOptionBuilder();

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
        [SweepableParameter(1E-4F, 10f, true, 20)]
        public float L2Regularization;

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
        [SweepableParameter(1E-4F, 10f, true, 20)]
        public float L1Regularization;

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
        [SweepableParameter(1, 1000, true, 20)]
        public int HistorySize;

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
    }
}
