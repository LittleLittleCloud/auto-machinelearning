// <copyright file="Parameters.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace MLNet.Sweeper
{
    /// <summary>
    /// A set of parameter values.
    /// The parameter set must be immutable.
    /// </summary>
    public sealed class Parameters : IEquatable<Parameters>, IEnumerable<IParameterValue>
    {
        private readonly Dictionary<string, IParameterValue> _parameterValues;
        private readonly int _hash;

        public Parameters(IEnumerable<IParameterValue> parameters)
        {
            this._parameterValues = new Dictionary<string, IParameterValue>();
            foreach (var parameter in parameters)
            {
                this._parameterValues.Add(parameter.ID, parameter);
            }

            var parameterNames = this._parameterValues.Keys.ToList();
            parameterNames.Sort();
            this._hash = 0;

            foreach (var parameterName in parameterNames)
            {
                this._hash = this._hash ^ this._parameterValues[parameterName].GetHashCode();
            }
        }

        public Parameters(Dictionary<string, IParameterValue> paramValues, int hash)
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

        public Dictionary<string, string> ParameterValues
        {
            get
            {
                var dict = new Dictionary<string, string>();
                foreach (var val in this)
                {
                    dict.Add(val.ID, val.ValueText);
                }

                return dict;
            }
        }

        public IParameterValue this[string id]
        {
            get
            {
                return this._parameterValues[id];
            }

            set
            {
                this._parameterValues[id] = value;
            }
        }

        public bool Equals(Parameters other)
        {
            if (other == null || other._hash != this._hash || other._parameterValues.Count != this._parameterValues.Count)
            {
                return false;
            }

            return other._parameterValues.Values.All(pv => this.ContainsParamValue(pv));
        }

        public Parameters Clone()
        {
            return new Parameters(new Dictionary<string, IParameterValue>(this._parameterValues), this._hash);
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
            return this._parameterValues.TryGetValue(parameterValue.ID, out value) &&
                   parameterValue.Equals(value);
        }
    }
}
