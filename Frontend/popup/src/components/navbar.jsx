import React from "react"
import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import { Button } from "react-bootstrap";
function NavbarHome(){
    return(
        <Navbar expand="xl" fixed="top">
             <Container fluid>
                 <Navbar.Toggle aria-controls="responsive-navbar-nav" />
                 <Navbar.Collapse id="responsive-navbar-nav">

                 </Navbar.Collapse>
                 <Nav>
                    <Button variant="primary" size="xsm" active 
                    style={{backgroundColor:"#80230A",
                    boxShadow:" 0 4px 12px rgba(0, 0, 0, 0.3)",
                    BorderColor:"white"}}
                    >
                        View Carts
                    </Button>
                <Button variant="secondary" size="xsm" active style={{marginLeft:"10px", 
               }} >
                + 
                </Button>
            </Nav>


             </Container>

        </Navbar>
    )
}
export default NavbarHome