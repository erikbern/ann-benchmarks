package dk.itu.epbprtv0.types;

public interface TypeInfo<T> {
  T parse(String item);
  String unparse(T item);
}