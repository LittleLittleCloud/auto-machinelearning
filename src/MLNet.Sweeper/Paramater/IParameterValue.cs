using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;

namespace MLNet.Sweeper
{
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

        [JsonIgnore]
        object RawValue { get; }

        [JsonIgnore]
        string ID { get; }
    }

    /// <summary>
    /// Type safe version of the IParameterValue interface.
    /// </summary>
    public interface IParameterValue<out TValue> : IParameterValue
    {
        [JsonIgnore]
        TValue Value { get; }
    }

    public interface IDiscreteParameterValue : IParameterValue
    {
        [JsonIgnore]
        double[] OneHotEncode { get; }
    }

    public interface IDiscreteParameterValue<out TValue> : IParameterValue<TValue>, IDiscreteParameterValue
    {
    }
}
