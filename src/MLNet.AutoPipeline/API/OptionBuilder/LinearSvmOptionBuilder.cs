// <copyright file="LinearSvmOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class LinearSvmOptionBuilder : OptionBuilder<LinearSvmTrainer.Options>
    {
        /// <summary>
        /// The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.
        /// <para>Default sweeping configuration.</para>
        /// <list type="bullet">
        /// <item>
        /// <term>type</term>
        /// <description><c>Int</c></description>
        /// </item>
        /// <item>
        /// <term>minimum value</term>
        /// <description><c>10</c></description>
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
        public SweepableParameter NumberOfIterations = SweepableParameter.CreateInt32Parameter(10, 1000, true, 20);

        /// <summary>
        /// The name of the example weight column.
        /// </summary>
        [Parameter]
        public string ExampleWeightColumnName = default;

        internal static LinearSvmOptionBuilder Default = new LinearSvmOptionBuilder();
    }
}
