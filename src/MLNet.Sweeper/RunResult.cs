// <copyright file="RunResult.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;

namespace MLNet.Sweeper
{
    /// <summary>
    /// Simple implementation of IRunResult.
    /// </summary>
    public sealed class RunResult : IRunResult<double>
    {
        private readonly IDictionary<string, string> _parameters;
        private readonly double? _metricValue;
        private readonly bool _isMetricMaximizing;

        /// <summary>
        /// This switch changes the behavior of the CompareTo function, switching the greater than / less than
        /// behavior, depending on if it is set to True.
        /// </summary>
        public bool IsMetricMaximizing => this._isMetricMaximizing;

        public IDictionary<string, string> ParameterSet => this._parameters;

        public RunResult(IDictionary<string, string> parameters, double metricValue, bool isMetricMaximizing)
        {
            this._parameters = parameters;
            this._metricValue = metricValue;
            this._isMetricMaximizing = isMetricMaximizing;
        }

        public double MetricValue
        {
            get
            {
                if (this._metricValue == null)
                {
                    throw new Exception("Run result does not contain a metric");
                }

                return this._metricValue.Value;
            }
        }

        public int CompareTo(IRunResult other)
        {
            var otherTyped = other as RunResult;
            if (this._metricValue == otherTyped._metricValue)
            {
                return 0;
            }

            return this._isMetricMaximizing ^ (this._metricValue < otherTyped._metricValue) ? 1 : -1;
        }

        public bool HasMetricValue => this._metricValue != null;

        IComparable IRunResult.MetricValue => this.MetricValue;
    }
}
