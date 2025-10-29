"""
FastMCP MoCoPo Server
--------------------------------
Define un servidor  FastMCP sencillo, publicando las funciones 
temperaturaCiudad, mocopo_two_numbers, factorial_par


"""

from fastmcp import FastMCP

# Create server
mcp = FastMCP("MoCoPoFast1")


@mcp.tool(name="temperaturaCiudad", description="optiene la temperatura en ciudades de España.")
def temperaturaCiudad (city: str) -> int :
  """
  optiene la temperatura en ciudades de España. Esta funcion se conectaria a un servicio web que devuelve el resultado en tiempo real.
  Args:
    city (str): la ciudad a consultar
  Returns:
    int: la temperatura de la ciudad
  """
  # log ('temperaturaCiudad:',city)
  if city.lower()=='teruel': return 7
  if city.lower()=='cuenca': return 14
  return 20

@mcp.tool(name="mocopo_two_numbers", description="Mocopo de dos numeros. Es el nombre que hemos inventado para una operacion compleja entre dos numeros")
def mocopo_two_numbers(a: int, b: int) -> int:
  """
  Mocopo de dos numeros. Es el nombre que hemos inventado para una operacion compleja entre dos numeros
  Args:
    'a' int : El primer numero
    'b' int : El segundo numero
   
  Returns:
    int: el resultado de la operacion
  """
  # log ('mocopo:',a,b)  
  return (int(a) + int(b))+(int(a) * int(b))


@mcp.tool(name="factorial_par", description="FactorialPar de un numero es un tipo de operacion de factorizacion especial que he inventado.")
def factorial_par (a: int) -> int:
  """
  FactorialPar de un numero. Un tipo de factorizacion especial definido como a!=a*(a-3)!
  Args:
    'a' int : El  numero
      
  Returns:
    int: el resultado de la operacion
  """
  a=int(a)
  if a<=1: return 1
  return a*factorial_par(a-3)

# python E:/miteruel/mcp/MoCoPoFast1.py --transport sse --port 3000

if __name__ == "__main__":
    # FastMCP detecta automáticamente si ejecutar en STDIO o HTTP
    # STDIO: python servidor_mcp.py
    # HTTP: python servidor_mcp.py --transport sse
    mcp.run()
    # mcp.run(transport="http", host="127.0.0.1", port=3001, path="/mcp")