// <copyright file="ParamaterSet.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MLNet.Sweeper
{
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

        private bool ContainsParamValue(IParameterValue parameterValue)
        {
            IParameterValue value;
            return this._parameterValues.TryGetValue(parameterValue.Name, out value) &&
                   parameterValue.Equals(value);
        }
    }
}
