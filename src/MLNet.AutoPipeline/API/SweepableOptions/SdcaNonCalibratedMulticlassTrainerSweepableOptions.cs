// <copyright file="SdcaNonCalibratedMulticlassTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Sweepable option for <see cref="SdcaNonCalibratedMulticlassTrainer"/>.
    /// </summary>
    public class SdcaNonCalibratedMulticlassTrainerSweepableOptions : SweepableOption<SdcaNonCalibratedMulticlassTrainer.Options>
    {
        public static SdcaNonCalibratedMulticlassTrainerSweepableOptions Default = new SdcaNonCalibratedMulticlassTrainerSweepableOptions();

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
        [Parameter]
        public Parameter<float> L2Regularization = CreateFloatParameter(1E-4F, 10f, true, 20);

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
        [Parameter]
        public Parameter<float> L1Reegularization = CreateFloatParameter(1E-4F, 10f, true, 20);
    }
}
