package dk.itu.epbprtv0.types;

import dk.itu.epbprtv0.Tokeniser;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

public class IntInfo implements TypeInfo<List<Integer>> {
  public static final IntInfo INSTANCE = new IntInfo();

  @Override
  public List<Integer> parse(String item) {
    ArrayList<Integer> results = new ArrayList<Integer>();
    for (String s : Tokeniser.tokenise(item))
      results.add(Integer.parseInt(s));
    return results;
  }

  @Override
  public String unparse(List<Integer> item) {
    StringBuilder sb = new StringBuilder();
    Iterator<Integer> it = item.iterator();
    while (it.hasNext()) {
      sb.append(it.next().toString());
      if (it.hasNext())
        sb.append(' ');
    }
    return sb.toString();
  }

  public static void main(String[] args) {
    List<Integer> h = IntInfo.INSTANCE.parse("  0  1  2  3  4  5  ");
    System.out.println(IntInfo.INSTANCE.unparse(h));
  }
}