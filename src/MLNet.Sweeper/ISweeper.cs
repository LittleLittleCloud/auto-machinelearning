// <copyright file="ISweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Internal.Utilities;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MLNet.Sweeper
{
    public interface ISweeper
    {
        /// <summary>
        /// Returns between 0 and maxSweeps configurations to run.
        /// It expects a list of previous runs such that it can generate configurations that were not already tried.
        /// The list of runs can be null if there were no previous runs.
        /// Some smart sweepers can take advantage of the metric(s) that the caller computes for previous runs.
        /// </summary>
        /// <returns>ParameterSet.</returns>
        ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null);
    }

    /// <summary>
    /// This is the interface that each type of parameter sweep needs to implement
    /// </summary>
    public interface IValueGenerator
    {
        /// <summary>
        /// Given a value in the [0,1] range, return a value for this parameter.
        /// </summary>
        IParameterValue CreateFromNormalized(Double normalizedValue);

        /// <summary>
        /// Used mainly in grid sweepers, return the i-th distinct value for this parameter.
        /// </summary>
        IParameterValue this[int i] { get; }

        /// <summary>
        /// Used mainly in grid sweepers, return the count of distinct values for this parameter.
        /// </summary>
        int Count { get; }

        /// <summary>
        /// Returns the name of the generated parameter.
        /// </summary>
        string Name { get; }
    }

    public interface ISweepResultEvaluator<in TResults>
    {
        /// <summary>
        /// Return an IRunResult based on the results given as a TResults object.
        /// </summary>
        IRunResult GetRunResult(ParameterSet parameters, TResults results);
    }

    /// <summary>
    /// Parameter value generated from the sweeping.
    /// The parameter values must be immutable.
    /// Value is converted to string because the runner will usually want to construct a command line for TL.
    /// Implementations of this interface must also override object.GetHashCode() and object.Equals(object) so they are consistent
    /// with IEquatable.Equals(IParameterValue).
    /// </summary>
    public interface IParameterValue : IEquatable<IParameterValue>
    {
        string Name { get; }

        string ValueText { get; }

        object RawValue { get; }
    }

    /// <summary>
    /// Type safe version of the IParameterValue interface.
    /// </summary>
    public interface IParameterValue<out TValue> : IParameterValue
    {
        TValue Value { get; }
    }

    /// <summary>
    /// A set of parameter values.
    /// The parameter set must be immutable.
    /// </summary>
    public sealed class ParameterSet : IEquatable<ParameterSet>, IEnumerable<IParameterValue>
    {
        private readonly Dictionary<string, IParameterValue> _parameterValues;
        private readonly int _hash;

        public ParameterSet(IEnumerable<IParameterValue> parameters)
        {
            this._parameterValues = new Dictionary<string, IParameterValue>();
            foreach (var parameter in parameters)
            {
                this._parameterValues.Add(parameter.Name, parameter);
            }

            var parameterNames = this._parameterValues.Keys.ToList();
            parameterNames.Sort();
            this._hash = 0;

            foreach (var parameterName in parameterNames)
            {
                this._hash = this._hash ^ this._parameterValues[parameterName].GetHashCode();
            }
        }

        public ParameterSet(Dictionary<string, IParameterValue> paramValues, int hash)
        {
            this._parameterValues = paramValues;
            this._hash = hash;
        }

        public IEnumerator<IParameterValue> GetEnumerator()
        {
            return this._parameterValues.Values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public int Count => this._parameterValues.Count;

        public IParameterValue this[string name] => this._parameterValues[name];

        private bool ContainsParamValue(IParameterValue parameterValue)
        {
            IParameterValue value;
            return this._parameterValues.TryGetValue(parameterValue.Name, out value) &&
                   parameterValue.Equals(value);
        }

        public bool Equals(ParameterSet other)
        {
            if (other == null || other._hash != this._hash || other._parameterValues.Count != this._parameterValues.Count)
            {
                return false;
            }

            return other._parameterValues.Values.All(pv => this.ContainsParamValue(pv));
        }

        public ParameterSet Clone()
        {
            return new ParameterSet(new Dictionary<string, IParameterValue>(this._parameterValues), this._hash);
        }

        public override string ToString()
        {
            return string.Join(" ", this._parameterValues.Select(kvp => string.Format("{0}={1}", kvp.Value.Name, kvp.Value.ValueText)).ToArray());
        }

        public override int GetHashCode()
        {
            return this._hash;
        }
    }

    /// <summary>
    /// The result of a run.
    /// Contains the parameter set used, useful for the sweeper to not generate the same configuration multiple times.
    /// Also contains the result of a run and the metric value that is used by smart sweepers to generate new configurations
    /// that try to maximize this metric.
    /// </summary>
    public interface IRunResult : IComparable<IRunResult>
    {
        ParameterSet ParameterSet { get; }

        IComparable MetricValue { get; }

        bool IsMetricMaximizing { get; }
    }

    public interface IRunResult<T> : IRunResult
        where T : IComparable<T>
    {
        new T MetricValue { get; }
    }
}
